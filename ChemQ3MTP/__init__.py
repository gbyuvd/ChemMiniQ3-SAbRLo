# __init__.py
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .configuration_chemq3mtp import ChemQ3MTPConfig
from .modeling_chemq3mtp import ChemQ3MTPForCausalLM
from .FastChemTokenizerHF import FastChemTokenizerSelfies  

# Register the model
AutoConfig.register("chemq3_mtp", ChemQ3MTPConfig)
AutoModelForCausalLM.register(ChemQ3MTPConfig, ChemQ3MTPForCausalLM)

# Register the tokenizer
AutoTokenizer.register(ChemQ3MTPConfig, FastChemTokenizerSelfies)

__all__ = ["ChemQ3MTPConfig", "ChemQ3MTPForCausalLM", "FastChemTokenizerSelfies"]