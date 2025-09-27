# ========================
#  Train with NTP + MTP
#  Updated for ChemQ3MTP structure
#  by gbyuvd
# ========================

# train_withmtp.py
import sys
import os
# Add the current directory to Python path so it can find your modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import List, Union, Optional, Tuple, Dict, Any
from transformers.tokenization_utils_base import BatchEncoding
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from ranger21 import Ranger21
from tqdm.notebook import tqdm
from FastChemTokenizerHF import FastChemTokenizerSelfies
from ChemQ3MTP import ChemQ3MTPConfig, ChemQ3MTPForCausalLM  # This should now work
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import TrainerCallback
import datetime

# Clear cache functions
def clear_cache():
    """Clear PyTorch and CUDA caches"""
    print("Clearing PyTorch and CUDA caches...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("CUDA cache cleared")
    torch.backends.cudnn.benchmark = True  # Enable cuDNN optimization
    print("PyTorch cache cleared")

def clear_datasets_cache():
    """Clear datasets cache directory"""
    import shutil
    from datasets import disable_caching, enable_caching, get_cache_directory
    try:
        cache_dir = get_cache_directory()
        print(f"Clearing datasets cache at: {cache_dir}")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("Datasets cache cleared")
    except:
        print("Could not clear datasets cache (may not exist)")

# ==============================
# Clear caches before starting
# ==============================
clear_cache()
# clear_datasets_cache()

# ==============================
# Load external configuration
# ==============================
with open("config.json", "r") as f:
    CONFIG = json.load(f)

TRAINING_CFG = CONFIG["training"]
MODEL_CFG = {k: v for k, v in CONFIG.items() 
             if k not in ["training", "generation", "model_type", "architectures"]}
GENERATION_CFG = CONFIG.get("generation", {})

# Training params
BATCH_SIZE = TRAINING_CFG["batch_size"]
NUM_EPOCHS = TRAINING_CFG["num_epochs"]
LEARNING_RATE = TRAINING_CFG["learning_rate"]
WEIGHT_DECAY = TRAINING_CFG["weight_decay"]
GRAD_ACCUM_STEPS = TRAINING_CFG["gradient_accumulation_steps"]
TOKENIZE_BATCH_SIZE = TRAINING_CFG["tokenize_batch_size"]
TRAIN_SPLIT_RATIO = TRAINING_CFG["train_split_ratio"]
VAL_SPLIT_RATIO = TRAINING_CFG["val_split_ratio"]
TEST_SPLIT_RATIO = TRAINING_CFG["test_split_ratio"]
INCLUDE_FOR_METRICS = TRAINING_CFG.get("include_for_metrics", ["input_ids", "attention_mask", "labels"])
# ==============================

class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_file="training_losses.txt", with_timestamp=False):
        self.log_file = log_file
        self.with_timestamp = with_timestamp
        with open(self.log_file, "w") as f:
            if self.with_timestamp:
                f.write("time\tstep\tloss\teval_loss\n")
            else:
                f.write("step\tloss\teval_loss\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")

        with open(self.log_file, "a") as f:
            if self.with_timestamp:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ts}\t{step}\t{loss if loss is not None else ''}\t{eval_loss if eval_loss is not None else ''}\n")
            else:
                f.write(f"{step}\t{loss if loss is not None else ''}\t{eval_loss if eval_loss is not None else ''}\n")


class CheckpointEvery10PercentCallback(TrainerCallback):
    """
    Custom callback to save checkpoints at 10% intervals of total training progress
    """
    def __init__(self, save_dir, total_steps):
        self.save_dir = save_dir
        self.total_steps = total_steps
        self.checkpoint_intervals = []
        # Calculate steps for 10% intervals (10%, 20%, 30%, ..., 100%)
        for i in range(1, 11):
            checkpoint_step = int(total_steps * i * 0.1)
            self.checkpoint_intervals.append(checkpoint_step)
        self.saved_checkpoints = set()
        print(f"Checkpoint intervals: {self.checkpoint_intervals}")

    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        
        # Check if we've reached a 10% checkpoint
        for checkpoint_step in self.checkpoint_intervals:
            if current_step == checkpoint_step and checkpoint_step not in self.saved_checkpoints:
                checkpoint_dir = f"{self.save_dir}/checkpoint_10percent_{current_step}"
                print(f"Saving 10% progress checkpoint at step {current_step} to {checkpoint_dir}")
                
                # Save model and tokenizer
                model = kwargs.get('model')
                tokenizer = kwargs.get('processing_class')  # or kwargs.get('tokenizer')
                
                if model is not None:
                    model.save_pretrained(checkpoint_dir)
                if tokenizer is not None:
                    tokenizer.save_pretrained(checkpoint_dir)
                
                # Also save training state
                if hasattr(kwargs.get('trainer'), 'save_state'):
                    kwargs['trainer'].save_state()
                
                self.saved_checkpoints.add(checkpoint_step)
                print(f"Checkpoint saved at step {current_step} ({current_step/self.total_steps*100:.1f}% completion)")
                break  # Only save one checkpoint per step


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize function defined outside main to avoid closure issues"""
    batch_results = {"input_ids": [], "attention_mask": [], "labels": []}
    smiles_list = examples['SELFIES'] if isinstance(examples['SELFIES'], list) else [examples['SELFIES']]
    for smiles in smiles_list:
        tokenized = tokenizer(
            smiles,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
            add_special_tokens=True
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.copy()
        batch_results["input_ids"].append(input_ids)
        batch_results["attention_mask"].append(attention_mask)
        batch_results["labels"].append(labels)
    return batch_results


def main():
    # Clear cache at the beginning of main function too
    clear_cache()
    
    # --- Load the tokenizer ---
    tokenizer = FastChemTokenizerSelfies.from_pretrained("../selftok_core")

    out = tokenizer("[C] [=C] [Branch1]", return_tensors="pt")
    print(out.input_ids)
    print(out.attention_mask)
    out = out.to("cuda" if torch.cuda.is_available() else "cpu")
    print(out.input_ids.device)

    checkpoint_path = "./chunk-2"

    if os.path.isdir(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = ChemQ3MTPForCausalLM.from_pretrained(checkpoint_path)
        config = model.config
    else:
        print("No checkpoint found, initializing new model.")
        config = ChemQ3MTPConfig(
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **MODEL_CFG
        )
        model = ChemQ3MTPForCausalLM(config)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Enhanced model has {count_parameters(model):,} trainable parameters.")

    batch_size, seq_len = 2, 32
    dummy_input = torch.randint(
        low=0,
        high=len(tokenizer),
        size=(batch_size, seq_len),
        dtype=torch.long,
    )
    with torch.no_grad():
        outputs = model(dummy_input)
        logits = outputs.logits
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")

    print("Loading dataset...")
    # Load dataset without streaming
    dataset = load_dataset(
        'csv',
        data_files='../data/chunk_3.csv',
        split='train'
    )

    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Verify the correct file is loaded by checking first few samples
    print("First few samples from dataset:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i}: {sample}")
        if 'SELFIES' in sample:
            print(f"First SELFIES: {sample['SELFIES']}")
        break

    print("Shuffling and splitting dataset...")
    # Shuffle the entire dataset first
    dataset = dataset.shuffle(seed=42)

    # Calculate split sizes
    total_lines = len(dataset)
    test_size = int(TEST_SPLIT_RATIO * total_lines)
    val_size = int(VAL_SPLIT_RATIO * total_lines)
    train_size = total_lines - test_size - val_size

    print(f"Total samples: {total_lines}")
    print(f"Split sizes - train: {train_size}, val: {val_size}, test: {test_size}")

    # Create splits using select
    train_dataset = dataset.select(range(0, train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total_lines))

    print(f"Dataset split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Tokenize datasets using batched mapping with explicit parameters
    print("Tokenizing datasets...")
    
    # Define tokenize function with all parameters passed explicitly
    def tokenize_train(examples):
        return tokenize_function(examples, tokenizer, MODEL_CFG["max_position_embeddings"])
    
    def tokenize_val(examples):
        return tokenize_function(examples, tokenizer, MODEL_CFG["max_position_embeddings"])

    train_dataset = train_dataset.map(
        tokenize_train, 
        batched=True, 
        batch_size=TOKENIZE_BATCH_SIZE, 
        remove_columns=["SELFIES"],
        desc="Tokenizing train dataset"
    )
    val_dataset = val_dataset.map(
        tokenize_val, 
        batched=True, 
        batch_size=TOKENIZE_BATCH_SIZE, 
        remove_columns=["SELFIES"],
        desc="Tokenizing val dataset"
    )

    class EnhancedDataCollator:
        def __init__(self, tokenizer, pad_to_multiple_of=8):
            self.tokenizer = tokenizer
            self.pad_to_multiple_of = pad_to_multiple_of
        def __call__(self, features):
            max_length = max(len(f["input_ids"]) for f in features)
            if self.pad_to_multiple_of:
                max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for feature in features:
                input_ids = feature["input_ids"]
                attention_mask = feature["attention_mask"]
                labels = feature["labels"]
                padding_length = max_length - len(input_ids)
                padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                padded_attention_mask = attention_mask + [0] * padding_length
                padded_labels = labels + [-100] * padding_length
                batch["input_ids"].append(padded_input_ids)
                batch["attention_mask"].append(padded_attention_mask)
                batch["labels"].append(padded_labels)
            batch = {key: torch.tensor(values, dtype=torch.long) for key, values in batch.items()}
            return batch

    data_collator = EnhancedDataCollator(tokenizer, pad_to_multiple_of=8)

    def create_enhanced_optimizer(model_params):
        num_batches_per_epoch = len(train_dataset) // BATCH_SIZE
        optimizer_params = {
            'lr': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'use_adabelief': True,
            'use_cheb': False,
            'use_warmup': True,
            'use_madgrad': True,
            'num_epochs': NUM_EPOCHS,
            'using_gc': True,
            'warmdown_active': True,
            'num_batches_per_epoch': num_batches_per_epoch
        }
        return Ranger21(model_params, **optimizer_params)

    from torch.optim.lr_scheduler import LambdaLR
    class EnhancedCustomTrainer(Trainer):
        def create_optimizer(self):
            self.optimizer = create_enhanced_optimizer(self.model.parameters())
            return self.optimizer
        def create_scheduler(self, num_training_steps, optimizer=None):
            if optimizer is None:
                optimizer = self.optimizer
            self.lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
            return self.lr_scheduler
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    total_steps = steps_per_epoch * NUM_EPOCHS

    training_args = TrainingArguments(
        output_dir='./chemq3minipret',
        max_steps=total_steps,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        logging_dir='./gptlo-1',
        logging_strategy="steps",
        logging_steps=max(1, steps_per_epoch // 4),
        eval_strategy="steps",
        eval_steps=max(1, steps_per_epoch // 4),
        save_strategy="steps",
        save_steps=steps_per_epoch,  # Save every epoch
        save_total_limit=1,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        prediction_loss_only=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        report_to=None,
        include_for_metrics=INCLUDE_FOR_METRICS,
    )

    print("Initializing enhanced trainer with MTP capabilities...")
    trainer = EnhancedCustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[
            LossLoggerCallback("training_losses.txt", with_timestamp=True),
            CheckpointEvery10PercentCallback("./chemq3minipret", total_steps)
        ]
    )

    model.set_mtp_training(True)
    print(" MTP training mode enabled")

    print("Starting enhanced training with MTP and Horizon Loss...")
    try:
        print("\n Phase 1: Warmup with standard Causal LM...")
        model.set_mtp_training(False)
        warmup_steps = max(1, total_steps // 5)
        
        # Update trainer args for warmup phase
        trainer.args.max_steps = warmup_steps
        trainer.train()
        print(f"\n Phase 1 completed. Warmup with {warmup_steps} steps finished.")
        
        print(f"\n Phase 2: Full MTP + Horizon Loss training...")
        print(f"Total training steps: {total_steps}")
        print(f"Training will save checkpoints at 10% intervals:")
        for i in range(1, 11):
            checkpoint_step = int(total_steps * i * 0.1)
            print(f"  - {i*10}%: Step {checkpoint_step}")
        
        model.set_mtp_training(True)
        # Reset max steps to total for the full training phase
        trainer.args.max_steps = total_steps
        trainer.train(resume_from_checkpoint=True)
        print("Enhanced training completed successfully!")
        trainer.save_model("./enhanced-qwen3-final")
        tokenizer.save_pretrained("./enhanced-qwen3-final")
        training_config = {
            "model_type": "ChemQ3MTPForCausalLM",
            "num_future_tokens": 3,
            "horizon_loss_enabled": True,
            "mtp_head_enabled": True,
            "training_phases": ["causal_lm_warmup", "mtp_horizon_training"],
            "total_parameters": count_parameters(model),
        }
        config_path = "./enhanced-qwen3-final/training_config.json"
        with open(config_path, "w") as f:
            json.dump(training_config, f, indent=2)
        print(f" Enhanced model, tokenizer, and config saved!")
    except Exception as e:
        print(f"Enhanced training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nmTesting enhanced generation capabilities...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    try:
        print("\n--- Standard Generation Test ---")
        input_ids = tokenizer("<s> [C]", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model.set_mtp_training(False)
            gen = model.generate(
                input_ids,
                max_length=25,
                top_k=50,
                top_p=0.9,
                temperature=1.0,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=3,
            )
            for i, sequence in enumerate(gen):
                result = tokenizer.decode(sequence, skip_special_tokens=True)
                print(f"Generated SELFIES {i+1}: {result}")
        print("\n--- MTP Analysis Test ---")
        test_smiles = "[C]"
        test_input = tokenizer(test_smiles, return_tensors="pt", add_special_tokens=True).to(device)
        test_input = {k: v for k, v in test_input.items() if k != 'token_type_ids'}  # Remove token_type_ids
        with torch.no_grad():
            outputs = model(**test_input)
            if hasattr(model, 'mtp_head') and hasattr(model.mtp_head, 'prediction_heads'):
                hidden_states = model.model(test_input['input_ids']).last_hidden_state
                mtp_outputs = model.mtp_head(hidden_states)
                print(f"Input SELFIES: {test_smiles}")
                print(f"Tokenized: {tokenizer.convert_ids_to_tokens(test_input['input_ids'][0].tolist())}")
                for i, (key, logits) in enumerate(mtp_outputs.items()):
                    top_tokens = torch.topk(logits[0], k=3, dim=-1)
                    print(f"\n{key} predictions:")
                    for pos in range(min(5, logits.size(1))):
                        pos_preds = []
                        for j in range(3):
                            token_id = top_tokens.indices[pos, j].item()
                            prob = torch.softmax(logits[0, pos], dim=-1)[token_id].item()
                            token = tokenizer.id_to_token.get(token_id, '<UNK>')
                            pos_preds.append(f"{token}({prob:.3f})")
                        print(f"  Position {pos}: {', '.join(pos_preds)}")
        print("\nEnhanced generation tests completed!")
    except Exception as e:
        print(f"Enhanced generation test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nEnhanced Model Analysis:")
    print(f"Total parameters: {count_parameters(model):,}")
    mtp_params = sum(p.numel() for p in model.mtp_head.parameters() if p.requires_grad)
    horizon_params = sum(p.numel() for p in model.horizon_loss.parameters() if p.requires_grad)
    base_params = count_parameters(model) - mtp_params - horizon_params
    print(f"Base model parameters: {base_params:,}")
    print(f"MTP head parameters: {mtp_params:,}")
    print(f"Horizon loss parameters: {horizon_params:,}")
    print(f"Enhancement overhead: {((mtp_params + horizon_params) / base_params * 100):.2f}%")
    print(f"\n Enhanced Model Architecture:")
    print(f"- Base Model: Qwen2 with {config.num_hidden_layers} layers")  # Updated this line
    print(f"- Hidden Size: {config.hidden_size}")
    print(f"- Attention Heads: {config.num_attention_heads}")
    print(f"- Vocab Size: {config.vocab_size}")
    print(f"- MTP Future Tokens: {model.mtp_head.num_future_tokens}")
    print(f"- Horizon Loss Weights: Learnable")
    print(f"- Training Mode: {'MTP + Horizon Loss' if model.use_mtp_training else 'Standard Causal LM'}")
    print("\n Enhanced training pipeline completed successfully!")

if __name__ == "__main__":
    main()