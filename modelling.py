# ========================
#  ChemQ3-MTP - HuggingFace Compatible Version
#  MODEL COMPONENTS 
#  by gbyuvd
# ========================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Union, Optional, Tuple, Dict, Any
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import selfies as sf
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import json
import numpy as np
from collections import Counter
from rdkit.Chem import rdMolDescriptors

logger = logging.get_logger(__name__)

# ========================
# CONFIGURATION CLASS
# ========================

class ChemQ3MTPConfig(Qwen2Config):
    """
    Configuration class for ChemQ3MTP model.
    """
    model_type = "chemq3_mtp"
    
    def __init__(
        self,
        num_future_tokens: int = 3,
        horizon_weights: Optional[List[float]] = None,
        use_mtp_training: bool = True,
        entropy_controller_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_future_tokens = num_future_tokens
        self.horizon_weights = horizon_weights or [0.9 ** i for i in range(num_future_tokens)]
        self.use_mtp_training = use_mtp_training
        self.entropy_controller_config = entropy_controller_config or {
            "min_entropy": 0.5,
            "max_entropy": 3.0,
            "target_entropy": 1.5,
            "adaptation_rate": 0.01
        }

# ========================
# UTILITY FUNCTIONS (kept minimal for HF compatibility)
# ========================

def selfies_to_smiles(selfies_str: str) -> str | None:
    """Convert SELFIES string to SMILES, handling tokenizer artifacts."""
    try:
        clean_selfies = selfies_str.replace(" ", "")
        return sf.decoder(clean_selfies)
    except Exception:
        return None

def is_valid_smiles(smiles: str) -> bool:
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return False
    return Chem.MolFromSmiles(smiles.strip()) is not None

# ========================
# MODEL COMPONENTS
# ========================

class MTPHead(nn.Module):
    """Multi-Token Prediction Head for predicting future tokens."""
    
    def __init__(self, hidden_size: int, vocab_size: int, num_future_tokens: int = 3):
        super().__init__()
        self.num_future_tokens = num_future_tokens
        self.vocab_size = vocab_size
        self.prediction_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(num_future_tokens)
        ])
        self.position_embeddings = nn.Embedding(num_future_tokens, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        outputs = {}
        
        for i in range(self.num_future_tokens):
            pos_emb = self.position_embeddings(torch.tensor(i, device=hidden_states.device))
            enhanced_hidden = self.layer_norm(hidden_states + pos_emb)
            logits = self.prediction_heads[i](enhanced_hidden)
            outputs[f'logits_t{i+1}'] = logits
            
        return outputs


class HorizonLoss(nn.Module):
    """Loss function for multi-horizon prediction."""
    
    def __init__(self, num_future_tokens: int = 3, horizon_weights: Optional[List[float]] = None):
        super().__init__()
        self.num_future_tokens = num_future_tokens
        if horizon_weights is None:
            self.horizon_weights = [0.9 ** i for i in range(num_future_tokens)]
        else:
            self.horizon_weights = horizon_weights
        self.log_weights = nn.Parameter(torch.log(torch.tensor(self.horizon_weights)))
        
    def forward(
        self, 
        mtp_outputs: Dict[str, torch.Tensor], 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        weights = F.softmax(self.log_weights, dim=0)
        total_loss = 0.0
        horizon_losses = {}
        
        for i in range(self.num_future_tokens):
            logits_key = f'logits_t{i+1}'
            if logits_key not in mtp_outputs:
                continue
                
            logits = mtp_outputs[logits_key]
            shift = i + 1
            if seq_len <= shift:
                continue
                
            shifted_logits = logits[:, :-shift, :].contiguous()
            shifted_targets = input_ids[:, shift:].contiguous()
            
            if attention_mask is not None:
                shifted_mask = attention_mask[:, shift:].contiguous()
                mask_expanded = shifted_mask.view(-1)
                valid_indices = mask_expanded == 1
                if valid_indices.sum() == 0:
                    continue
                flat_logits = shifted_logits.view(-1, logits.size(-1))[valid_indices]
                flat_targets = shifted_targets.view(-1)[valid_indices]
            else:
                flat_logits = shifted_logits.view(-1, logits.size(-1))
                flat_targets = shifted_targets.view(-1)
                
            horizon_loss = F.cross_entropy(flat_logits, flat_targets, reduction='mean')
            horizon_losses[f'horizon_loss_t{i+1}'] = horizon_loss
            total_loss += weights[i] * horizon_loss
            
        return {'loss': total_loss, 'horizon_weights': weights, **horizon_losses}


class EnhancedEntropyController:
    """Enhanced entropy controller for adaptive training."""
    
    def __init__(self, min_entropy: float = 0.5, max_entropy: float = 3.0,
                 target_entropy: float = 1.5, adaptation_rate: float = 0.01):
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy
        self.target_entropy = target_entropy
        self.adaptation_rate = adaptation_rate
        self.entropy_history = []
        self.entropy_weight = 0.01
        
    def update_entropy_weight(self, current_entropy: float) -> float:
        """Dynamically adjust entropy weight based on current entropy levels."""
        self.entropy_history.append(current_entropy)
        
        if len(self.entropy_history) > 100:
            self.entropy_history = self.entropy_history[-100:]
            
        if len(self.entropy_history) >= 10:
            avg_entropy = np.mean(self.entropy_history[-10:])
            
            if avg_entropy < self.target_entropy * 0.8:
                self.entropy_weight = min(0.05, self.entropy_weight * 1.1)
            elif avg_entropy > self.target_entropy * 1.2:
                self.entropy_weight = max(0.001, self.entropy_weight * 0.95)
                
        return self.entropy_weight

# ========================
# MAIN MODEL CLASS
# ========================

class ChemQ3MTPForCausalLM(Qwen2ForCausalLM):
    """
    ChemQ3MTP model for causal language modeling with multi-token prediction.
    
    This model extends Qwen2ForCausalLM with additional capabilities for
    multi-token prediction and chemistry-specific training.
    """
    
    config_class = ChemQ3MTPConfig
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    
    def __init__(self, config: ChemQ3MTPConfig):
        super().__init__(config)
        
        # Initialize MTP components
        self.mtp_head = MTPHead(
            config.hidden_size, 
            config.vocab_size, 
            config.num_future_tokens
        )
        self.horizon_loss = HorizonLoss(
            num_future_tokens=config.num_future_tokens,
            horizon_weights=config.horizon_weights
        )
        
        # Training configuration
        self.use_mtp_training = config.use_mtp_training
        
        # Initialize entropy controller
        self.entropy_controller = EnhancedEntropyController(
            **config.entropy_controller_config
        )
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of the ChemQ3MTP model.
        """
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Default attention mask if not provided
        if attention_mask is None and input_ids is not None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        # Call parent forward with required hidden states
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,  # Handle labels manually
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always need hidden states for MTP
            return_dict=True,
            cache_position=cache_position,
        )

        hidden_states = outputs.hidden_states[-1]
        lm_logits = outputs.logits
        loss = None

        # Compute loss if labels are provided
        if labels is not None:
            if self.training and self.use_mtp_training:
                # Multi-token prediction training
                mtp_outputs = self.mtp_head(hidden_states)
                horizon_loss_dict = self.horizon_loss(mtp_outputs, input_ids, attention_mask)

                # Standard causal LM loss
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                if attention_mask is not None:
                    shift_mask = attention_mask[..., 1:].contiguous()
                    loss_mask = shift_mask.view(-1) == 1
                    if loss_mask.sum() == 0:
                        causal_lm_loss = torch.tensor(0.0, device=lm_logits.device)
                    else:
                        flat_logits = shift_logits.view(-1, shift_logits.size(-1))[loss_mask]
                        flat_labels = shift_labels.view(-1)[loss_mask]
                        causal_lm_loss = F.cross_entropy(flat_logits, flat_labels, reduction='mean')
                else:
                    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                    flat_labels = shift_labels.view(-1)
                    causal_lm_loss = F.cross_entropy(flat_logits, flat_labels, reduction='mean')

                # Combine losses
                loss = 0.7 * horizon_loss_dict['loss'] + 0.3 * causal_lm_loss

            else:
                # Standard causal LM training
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def set_mtp_training(self, use_mtp: bool):
        """Enable or disable multi-token prediction training."""
        self.use_mtp_training = use_mtp

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs
    ):
        """
        Prepare inputs for generation. This method is required for compatibility
        with HuggingFace's generation utilities.
        """
        # This delegates to the parent class implementation
        return super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs
        )

    def generate_with_logprobs(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        return_probs: bool = True,
        tokenizer=None,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate sequences with log probabilities for RL training.
        """
        self.eval()
        device = input_ids.device

        # Normalize input shapes
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() == 3 and input_ids.size(1) == 1:
            input_ids = input_ids.squeeze(1)
        assert input_ids.dim() == 2, f"input_ids must be 2-D, got {input_ids.shape}"

        batch_size, seq_len = input_ids.shape
        current_input = input_ids

        generated_tokens, generated_logprobs, generated_probs = [], [], []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self(current_input, use_cache=False)
                logits = outputs.logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    values, indices = torch.topk(logits, k=top_k)
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(1, indices, values)

                # Apply top-p filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumprobs > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    logits[mask.scatter(1, sorted_indices, mask)] = float("-inf")

                probs = F.softmax(logits, dim=-1)

                if do_sample:
                    dist = Categorical(probs)
                    next_token = dist.sample()
                    log_p = dist.log_prob(next_token)
                else:
                    next_token = torch.argmax(probs, dim=-1)
                    log_p = torch.log(torch.gather(probs, 1, next_token.unsqueeze(1))).squeeze(1)

                generated_tokens.append(next_token.unsqueeze(1))
                generated_logprobs.append(log_p.unsqueeze(1))
                if return_probs:
                    generated_probs.append(probs.unsqueeze(1))

                current_input = torch.cat([current_input, next_token.unsqueeze(1)], dim=1)

        generated_tokens = torch.cat(generated_tokens, dim=1)
        generated_logprobs = torch.cat(generated_logprobs, dim=1)
        generated_probs = torch.cat(generated_probs, dim=1) if return_probs else None

        # Decode generated tokens
        if tokenizer is None:
            tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided to decode generated tokens.")

        decoded_list = [
            tokenizer.decode(tok_ids, skip_special_tokens=True)
            for tok_ids in generated_tokens
        ]
        
        return decoded_list, generated_logprobs, generated_tokens, generated_probs

# ========================
# REGISTRATION
# ========================
from .configuration_chemq3mtp import ChemQ3MTPConfig

# Register the configuration and model classes
from transformers import AutoConfig, AutoModel

AutoConfig.register("chemq3_mtp", ChemQ3MTPConfig)
AutoModel.register(ChemQ3MTPConfig, ChemQ3MTPForCausalLM)