# test_model.py
import json
from configuration_chemq3mtp import ChemQ3MTPConfig
from modeling_chemq3mtp import ChemQ3MTPForCausalLM
import torch

# Load your config from the JSON file
with open("config.json", "r") as f:
    config_data = json.load(f)

# Create the model config using the top-level parameters
config = ChemQ3MTPConfig.from_dict(config_data)

model = ChemQ3MTPForCausalLM(config)

# Test basic functionality
dummy_input = torch.randint(0, config.vocab_size, (2, 10))
outputs = model(input_ids=dummy_input)
print(f"Model output shape: {outputs.logits.shape}")
print("✅ Model loaded and working!")

# Test saving and loading
model.save_pretrained("./test_model")

# Now test with AutoModel
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("./test_model")
model = AutoModelForCausalLM.from_pretrained("./test_model")
print("✅ AutoModel loaded successfully!")