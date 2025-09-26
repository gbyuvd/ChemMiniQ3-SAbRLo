import torch
import sys
import os

# Add the current directory to path to find your modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# CRITICAL: Import your custom modules FIRST to trigger registration
from ChemQ3MTP import ChemQ3MTPConfig, ChemQ3MTPForCausalLM, FastChemTokenizerSelfies

# Now test Auto classes loading from the same directory
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# --- Load using Auto classes ---
print("Loading with Auto classes...")

MODEL_DIR = './chemq3minipret/checkpoint_10percent_121'
# This should automatically use your custom model and tokenizer
config = AutoConfig.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print(f"✅ Config loaded: {config.model_type}")
print(f"✅ Model loaded: {type(model).__name__}")
print(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")

# Test the tokenizer
print("\n--- Tokenizer Test ---")
out = tokenizer("[C]", return_tensors="pt")
print(f"Tokenized input: {out.input_ids}")
print(f"Attention mask: {out.attention_mask}")
print(f"Device: {out.input_ids.device}")

# Test model
print("\n--- Model Test ---")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model has {count_parameters(model):,} trainable parameters.")

# Quick forward pass
batch_size, seq_len = 2, 32
dummy_input = torch.randint(
    low=0,
    high=len(tokenizer),
    size=(batch_size, seq_len),
    dtype=torch.long,
).to(model.device)

with torch.no_grad():
    outputs = model(dummy_input)
    logits = outputs.logits

print(f"Input shape: {dummy_input.shape}")
print(f"Logits shape: {logits.shape}")

# Test generation
print("\n--- Generation Test ---")
input_ids = tokenizer("[C]", return_tensors="pt").input_ids.to(model.device)
with torch.no_grad():
    generated = model.generate(
        input_ids,
        max_length=20,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    result = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated: {result}")

print("✅ Auto loading test completed successfully!")

# Generate SELFIES
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_ids = tokenizer("<s>", return_tensors="pt").input_ids.to(device)
gen = model.generate(input_ids, max_length=256, top_k=50, temperature=1, do_sample=True, pad_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(gen[0], skip_special_tokens=True))

# Manually convert it to SMILES
import selfies as sf

test = tokenizer.decode(gen[0], skip_special_tokens=True)
test = test.replace(' ', '')
print(sf.decoder(test))

# Generate Mol Viz
from rdkit import Chem
from rdkit.Chem import Draw

input_ids = tokenizer("<s>", return_tensors="pt").input_ids.to(device)
gen = model.generate(input_ids, max_length=256, top_k=50, temperature=1, do_sample=True, pad_token_id=tokenizer.pad_token_id)
generatedmol = tokenizer.decode(gen[0], skip_special_tokens=True)

test = generatedmol.replace(' ', '')
csmi_gen = sf.decoder(test)
print(csmi_gen)
mol = Chem.MolFromSmiles(csmi_gen)

# Draw the molecule
Draw.MolToImage(mol)