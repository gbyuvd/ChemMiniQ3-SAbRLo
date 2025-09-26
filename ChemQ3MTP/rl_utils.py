# ========================
#  RL_UTILS.PY
#  Chemistry RL Training Utilities for ChemQ3-MTP
#  by gbyuvd
# ========================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Union, Optional, Tuple, Dict, Any
import numpy as np
from collections import Counter

# Chemistry imports
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
import selfies as sf
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Optional: HuggingFace for SA classifier
try:
    from transformers import pipeline, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers not available, SA classifier will not work")

# ========================
# CHEMISTRY UTILITIES
# ========================

def selfies_to_smiles(selfies_str: str) -> str | None:
    """Convert SELFIES string to SMILES, handling tokenizer artifacts."""
    try:
        clean_selfies = selfies_str.replace(" ", "")
        return sf.decoder(clean_selfies)
    except Exception:
        return None

def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string represents a valid molecule."""
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return False
    return Chem.MolFromSmiles(smiles.strip()) is not None

# ========================
# SA CLASSIFIER
# ========================

# Global classifier instance for lazy loading
_sa_classifier = None

def get_sa_classifier():
    """Get or initialize the synthetic accessibility classifier."""
    global _sa_classifier
    if not HF_AVAILABLE:
        raise ImportError("transformers package required for SA classifier")
    
    if _sa_classifier is None:
        try:
            sa_tokenizer = AutoTokenizer.from_pretrained("gbyuvd/synthaccess-chemselfies")
            _sa_classifier = pipeline(
                "text-classification",
                model="gbyuvd/synthaccess-chemselfies",
                tokenizer=sa_tokenizer
            )
        except Exception as e:
            print(f"Warning: Could not load SA classifier: {e}")
            return None
    return _sa_classifier

def compute_sa_reward(selfies_str: str) -> float:
    """Reward molecules with easy synthetic accessibility (SA)."""
    try:
        classifier = get_sa_classifier()
        if classifier is None:
            return 0.0
        
        result = classifier(selfies_str, truncation=True, max_length=128)[0]
        if result["label"].lower() == "easy":
            return result["score"]
        else:
            return -result["score"]  # penalize "Hard"
    except Exception:
        return 0.0

# ========================
# MOLECULAR REWARD COMPONENTS
# ========================

def compute_biological_diversity_score(mol) -> float:
    """Reward molecules with diverse CHONP atoms, normalized to [0,1]."""
    if mol is None:
        return 0.0
    try:
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atom_counts = Counter(atoms)
        bio_elements = {"C", "H", "O", "N", "P"}
        present_bio_elements = set(atoms) & bio_elements

        if len(present_bio_elements) < 2:
            return 0.0

        base_score = 0.3
        diversity_bonus = (len(present_bio_elements) - 2) / 3 * 0.4

        total_bio_atoms = sum(atom_counts.get(e, 0) for e in present_bio_elements)
        if total_bio_atoms > 0:
            bio_probs = [atom_counts.get(e, 0) / total_bio_atoms for e in present_bio_elements]
            if len(bio_probs) > 1:
                entropy = -sum(p * np.log2(p) for p in bio_probs if p > 0)
                max_entropy = np.log2(len(bio_probs))
                entropy_bonus = (entropy / max_entropy) * 0.3
            else:
                entropy_bonus = 0.0
        else:
            entropy_bonus = 0.0

        return min(1.0, base_score + diversity_bonus + entropy_bonus)
    except Exception:
        return 0.0

def compute_charge_neutrality_score(mol) -> float:
    """Reward if molecule is globally neutral (formal charge = 0)."""
    if mol is None:
        return 0.0
    try:
        return 1.0 if Chem.rdmolops.GetFormalCharge(mol) == 0 else 0.0
    except Exception:
        return 0.0

def compute_local_charge_penalty(mol) -> float:
    """
    Penalize carbocations/anions.
    Returns 1.0 if no charged atoms, decreases with fraction charged.
    """
    if mol is None:
        return 0.0
    try:
        charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        if not charges:
            return 1.0
        charged_atoms = sum(1 for c in charges if c != 0)
        total_atoms = len(charges)
        return max(0.0, 1.0 - (charged_atoms / total_atoms))
    except Exception:
        return 0.0

def compute_enhanced_lipinski_reward(mol) -> float:
    """Soft Lipinski scoring with partial credit."""
    if mol is None:
        return 0.0
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        scores = []

        # Molecular Weight
        if 250 <= mw <= 500:
            scores.append(1.0)
        elif 150 <= mw < 250:
            scores.append(0.5)
        elif 500 < mw <= 600:
            scores.append(0.7)
        else:
            scores.append(0.0)

        # LogP
        if -1 <= logp <= 5:
            scores.append(1.0)
        elif -2 <= logp < -1 or 5 < logp <= 6:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # Hydrogen bond donors
        scores.append(1.0 if hbd <= 5 else max(0.0, 1.0 - 0.2 * (hbd - 5)))
        
        # Hydrogen bond acceptors
        scores.append(1.0 if hba <= 10 else max(0.0, 1.0 - 0.1 * (hba - 10)))

        return sum(scores) / len(scores)
    except Exception:
        return 0.0

def compute_structural_complexity_reward(mol) -> float:
    """Reward moderate complexity: 1–3 rings and some flexibility."""
    if mol is None:
        return 0.0
    try:
        ring_count = rdMolDescriptors.CalcNumRings(mol)
        if 1 <= ring_count <= 3:
            ring_score = 1.0
        elif ring_count == 0:
            ring_score = 0.3
        elif ring_count <= 5:
            ring_score = 0.7
        else:
            ring_score = 0.1

        rot_bonds = Descriptors.NumRotatableBonds(mol)
        if 2 <= rot_bonds <= 8:
            flex_score = 1.0
        elif rot_bonds <= 12:
            flex_score = 0.7
        elif rot_bonds in (0, 1):
            flex_score = 0.5
        else:
            flex_score = 0.2

        return (ring_score + flex_score) / 2
    except Exception:
        return 0.0

def compute_lipinski_reward(mol) -> float:
    """Simple Lipinski rule compliance scoring."""
    if mol is None:
        return 0.0
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        # We don't want too small fragments, so MW > 250
        rules = [250 < mw <= 500, logp <= 5, hbd <= 5, hba <= 10]
        return sum(rules) / 4.0
    except Exception:
        return 0.0

# ========================
# COMPREHENSIVE REWARD SYSTEM
# ========================

def compute_comprehensive_reward(selfies_str: str) -> Dict[str, float]:
    """
    Compute comprehensive reward for a SELFIES string.
    
    Args:
        selfies_str: SELFIES representation of molecule
        
    Returns:
        Dictionary containing individual reward components and total
    """
    smiles = selfies_to_smiles(selfies_str)
    mol = Chem.MolFromSmiles(smiles) if smiles else None

    rewards = {
        "validity": 1.0 if mol is not None else 0.0,
        "biological_diversity": compute_biological_diversity_score(mol),
        "charge_neutrality": compute_charge_neutrality_score(mol),
        "local_charge_penalty": compute_local_charge_penalty(mol),
        "lipinski": compute_enhanced_lipinski_reward(mol),
        "structural_complexity": compute_structural_complexity_reward(mol),
    }

    if rewards["validity"] == 0:
        rewards["total"] = 0.0
    else:
        # Weighted combination of rewards
        weights = {
            "validity": 1.0,
            "biological_diversity": 2.0,
            "charge_neutrality": 1.5,
            "local_charge_penalty": 1.0,
            "lipinski": 1.0,
            "structural_complexity": 0.5,
        }
        weighted_sum = sum(rewards[k] * weights[k] for k in weights)
        rewards["total"] = weighted_sum / sum(weights.values())

    return rewards

def selfies_to_lipinski_reward(selfies_str: str) -> float:
    """Convert SELFIES to SMILES, then compute Lipinski reward."""
    smiles = selfies_to_smiles(selfies_str)
    if smiles is None:
        return 0.0
    mol = Chem.MolFromSmiles(smiles)
    return compute_lipinski_reward(mol)

# ========================
# RL TRAINING CONTROLLERS
# ========================

class AdaptiveKLController:
    """
    Adaptive KL divergence controller for PPO training.
    Increases or decreases β so that E[KL] stays ≈ target_kl.
    """
    
    def __init__(
        self, 
        init_kl_coef: float = 0.1, 
        target_kl: float = 0.01,
        kl_horizon: int = 1000, 
        increase_rate: float = 1.5, 
        decrease_rate: float = 0.8
    ):
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.kl_horizon = kl_horizon
        self.inc = increase_rate
        self.dec = decrease_rate
        self.buffer = []

    def update(self, kl: float) -> float:
        """Update KL coefficient based on observed KL divergence."""
        self.buffer.append(kl)
        
        if len(self.buffer) >= self.kl_horizon:
            avg_kl = sum(self.buffer) / len(self.buffer)
            self.buffer.clear()
            
            if avg_kl > self.target_kl * 1.5:
                self.kl_coef *= self.inc
                print(f"KL too high ({avg_kl:.4f}), increasing β to {self.kl_coef:.4f}")
            elif avg_kl < self.target_kl * 0.5:
                self.kl_coef *= self.dec
                print(f"KL too low ({avg_kl:.4f}), decreasing β to {self.kl_coef:.4f}")
                
        return self.kl_coef

class EnhancedEntropyController:
    """
    Enhanced entropy controller with dynamic targets and temperature scheduling.
    """
    
    def __init__(
        self, 
        min_entropy: float = 0.5, 
        max_entropy: float = 3.0,
        target_entropy: float = 1.5, 
        adaptation_rate: float = 0.01
    ):
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy
        self.target_entropy = target_entropy
        self.adaptation_rate = adaptation_rate
        self.entropy_history = []
        self.entropy_weight = 0.01  # Starting weight
        
    def update_entropy_weight(self, current_entropy: float) -> float:
        """Dynamically adjust entropy weight based on current entropy levels."""
        self.entropy_history.append(current_entropy)
        
        # Keep rolling window
        if len(self.entropy_history) > 100:
            self.entropy_history = self.entropy_history[-100:]
            
        if len(self.entropy_history) >= 10:
            avg_entropy = np.mean(self.entropy_history[-10:])
            
            # If entropy too low, increase weight to encourage exploration
            if avg_entropy < self.target_entropy * 0.8:
                self.entropy_weight = min(0.05, self.entropy_weight * 1.1)
            # If entropy too high, decrease weight
            elif avg_entropy > self.target_entropy * 1.2:
                self.entropy_weight = max(0.001, self.entropy_weight * 0.95)
                
        return self.entropy_weight
    
    def compute_entropy_reward(self, entropy: float) -> float:
        """Reward function for entropy - prefer target range."""
        if self.min_entropy <= entropy <= self.max_entropy:
            # Gaussian reward centered at target
            distance = abs(entropy - self.target_entropy)
            max_distance = max(
                self.target_entropy - self.min_entropy, 
                self.max_entropy - self.target_entropy
            )
            return np.exp(-(distance / max_distance) ** 2)
        else:
            return 0.1  # Small penalty for being outside range

class CurriculumManager:
    """
    Curriculum learning manager for progressive training.
    Gradually increases max_new_tokens from start_len → max_len, then cycles.
    """
    
    def __init__(
        self, 
        start_len: int = 10, 
        max_len: int = 30, 
        step_increase: int = 5, 
        steps_per_level: int = 30
    ):
        self.start_len = start_len
        self.max_len = max_len
        self.step_increase = step_increase
        self.steps_per_level = steps_per_level
        self.step_counter = 0
        self.current_max_len = start_len

    def get_max_new_tokens(self) -> int:
        """Get current maximum new tokens."""
        return self.current_max_len

    def step(self) -> int:
        """Update curriculum and return new max_new_tokens."""
        self.step_counter += 1
        
        if self.step_counter % self.steps_per_level == 0:
            if self.current_max_len < self.max_len:
                self.current_max_len = min(
                    self.current_max_len + self.step_increase, 
                    self.max_len
                )
            else:
                # Reset cycle
                self.current_max_len = self.start_len
                print(f"🔄 Cycle reset: max_new_tokens -> {self.current_max_len}")
                
            if self.current_max_len < self.max_len:
                print(f"📈 Curriculum Update: max_new_tokens = {self.current_max_len}")
                
        return self.current_max_len

# ========================
# PPO TRAINING UTILITIES
# ========================

def compute_ppo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    clip_epsilon: float = 0.2,
    baseline: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute PPO clipped loss.
    
    Args:
        old_log_probs: Log probabilities from old policy [B, T]
        new_log_probs: Log probabilities from new policy [B, T]  
        rewards: Reward values [B]
        clip_epsilon: Clipping parameter
        baseline: Optional baseline for advantage computation [B]
        
    Returns:
        Tuple of (ppo_loss, advantage)
    """
    # Compute advantage
    if baseline is not None:
        advantage = rewards - baseline
    else:
        advantage = rewards
    
    # Probability ratio
    log_ratio = new_log_probs.sum(dim=1) - old_log_probs.sum(dim=1)  # [B]
    ratio = torch.exp(log_ratio)
    
    # PPO loss
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
    ppo_loss = -torch.min(surr1, surr2).mean()
    
    return ppo_loss, advantage

def compute_kl_divergence(
    old_action_probs: torch.Tensor,
    new_action_probs: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between old and new action distributions.
    
    Args:
        old_action_probs: Old action probabilities [B, T, V]
        new_action_probs: New action probabilities [B, T, V]
        
    Returns:
        KL divergence per example [B]
    """
    old_probs = old_action_probs.clamp_min(1e-12)
    new_probs = new_action_probs.clamp_min(1e-12)
    
    # KL(old || new) = sum(old * log(old / new))
    kl = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(dim=-1)  # [B, T]
    return kl.sum(dim=1)  # [B]

def compute_entropy_bonus(action_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy bonus for exploration.
    
    Args:
        action_probs: Action probabilities [B, T, V]
        
    Returns:
        Entropy per example [B]
    """
    probs = action_probs.clamp_min(1e-12)
    entropy = -(probs * torch.log(probs)).sum(dim=-1)  # [B, T]
    return entropy.sum(dim=1)  # [B]

# ========================
# BATCH REWARD COMPUTATION
# ========================

def batch_compute_rewards(
    selfies_list: List[str],
    reward_mode: str = "chemq3",
    reward_mix: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    Compute rewards for a batch of SELFIES strings.
    
    Args:
        selfies_list: List of SELFIES strings
        reward_mode: "chemq3", "sa", or "mix"
        reward_mix: Weight for chemq3 rewards when mixing (0-1)
        
    Returns:
        Dictionary containing reward tensors
    """
    batch_size = len(selfies_list)
    
    validity_vals = []
    lipinski_vals = []
    total_rewards = []
    sa_rewards = []

    for selfies_str in selfies_list:
        if reward_mode == "chemq3":
            r = compute_comprehensive_reward(selfies_str)
            validity_vals.append(r.get('validity', 0.0))
            lipinski_vals.append(r.get('lipinski', 0.0))
            total_rewards.append(r.get('total', 0.0))

        elif reward_mode == "sa":
            sa = compute_sa_reward(selfies_str)
            sa_rewards.append(sa)
            total_rewards.append(sa)

        elif reward_mode == "mix":
            r = compute_comprehensive_reward(selfies_str)
            sa = compute_sa_reward(selfies_str)
            mixed = reward_mix * r.get("total", 0.0) + (1.0 - reward_mix) * sa
            
            total_rewards.append(mixed)
            sa_rewards.append(sa)
            validity_vals.append(r.get('validity', 0.0))
            lipinski_vals.append(r.get('lipinski', 0.0))

        else:
            # Unknown mode -> default to zero reward
            total_rewards.append(0.0)
            validity_vals.append(0.0)
            lipinski_vals.append(0.0)

    # Convert to tensors
    result = {
        "total_rewards": torch.tensor(total_rewards, dtype=torch.float32),
    }
    
    if validity_vals:
        result["validity_rewards"] = torch.tensor(validity_vals, dtype=torch.float32)
    if lipinski_vals:
        result["lipinski_rewards"] = torch.tensor(lipinski_vals, dtype=torch.float32)
    if sa_rewards:
        result["sa_rewards"] = torch.tensor(sa_rewards, dtype=torch.float32)
        
    return result

# ========================
# TRAINING METRICS
# ========================

def compute_training_metrics(
    rewards: Dict[str, torch.Tensor],
    selfies_list: List[str],
    loss_dict: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute comprehensive training metrics.
    
    Args:
        rewards: Dictionary of reward tensors
        selfies_list: List of generated SELFIES
        loss_dict: Dictionary containing loss components
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Basic reward metrics
    if "total_rewards" in rewards:
        metrics["avg_reward"] = float(rewards["total_rewards"].mean())
        metrics["max_reward"] = float(rewards["total_rewards"].max())
        metrics["min_reward"] = float(rewards["total_rewards"].min())
        metrics["reward_std"] = float(rewards["total_rewards"].std())
    
    if "validity_rewards" in rewards:
        metrics["validity_rate"] = float(rewards["validity_rewards"].mean())
    
    if "lipinski_rewards" in rewards:
        metrics["lipinski_score"] = float(rewards["lipinski_rewards"].mean())
    
    if "sa_rewards" in rewards:
        metrics["sa_score"] = float(rewards["sa_rewards"].mean())
    
    # Molecular diversity metrics
    valid_smiles = []
    for selfies_str in selfies_list:
        smiles = selfies_to_smiles(selfies_str)
        if smiles and is_valid_smiles(smiles):
            valid_smiles.append(smiles)
    
    metrics["num_valid"] = len(valid_smiles)
    metrics["num_unique"] = len(set(valid_smiles))
    metrics["diversity_ratio"] = len(set(valid_smiles)) / max(1, len(valid_smiles))
    
    # Add loss components
    metrics.update(loss_dict)
    
    return metrics
