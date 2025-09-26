# test_model.py
import json
from configuration_chemq3mtp import ChemQ3MTPConfig
from modeling_chemq3mtp import ChemQ3MTPForCausalLM
from FastChemTokenizerHF import FastChemTokenizerSelfies
import torch

# Load your config from the JSON file
with open("config.json", "r") as f:
    config_data = json.load(f)

# Load the tokenizer to get the actual vocab size
tokenizer = FastChemTokenizerSelfies.from_pretrained("./")

# Update config with actual tokenizer parameters
config_data["vocab_size"] = len(tokenizer)
config_data["pad_token_id"] = tokenizer.pad_token_id
config_data["bos_token_id"] = tokenizer.bos_token_id
config_data["eos_token_id"] = tokenizer.eos_token_id

# Create the model config using the updated parameters
config = ChemQ3MTPConfig.from_dict(config_data)

model = ChemQ3MTPForCausalLM(config)

# Test basic functionality
dummy_input = torch.randint(0, config.vocab_size, (2, 10))
outputs = model(input_ids=dummy_input)
print(f"Model output shape: {outputs.logits.shape}")
print("✅ Model loaded and working!")

# Test with actual tokenizer input
test_text = "[C] [C] [O]"
tokenized = tokenizer(test_text, return_tensors="pt")
test_outputs = model(input_ids=tokenized.input_ids)
print(f"Test input shape: {tokenized.input_ids.shape}")
print(f"Test output shape: {test_outputs.logits.shape}")
print("✅ Model works with tokenizer!")

# Test saving and loading
model.save_pretrained("./test_model")
tokenizer.save_pretrained("./test_model")

# Now test with AutoModel
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("./test_model")
model = AutoModelForCausalLM.from_pretrained("./test_model")
tokenizer_loaded = FastChemTokenizerSelfies.from_pretrained("./test_model")

print("✅ AutoModel loaded successfully!")
print(f"Loaded vocab size: {config.vocab_size}")
print(f"Loaded tokenizer vocab size: {len(tokenizer_loaded)}")