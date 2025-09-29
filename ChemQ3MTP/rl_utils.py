# ========================
#  RL_UTILS.PY
#  v3
#  Chemistry RL Training Utilities for ChemQ3-MTP
#  by gbyuvd
#  Patched: reward normalization, KL/entropy reset per phase,
#           entropy target annealing, and symmetric curriculum
#           and now with Durrant's Lab's filtering included
# ========================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Union, Optional, Tuple, Dict, Any
import numpy as np
from collections import Counter, deque

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
    """
    Check if a SMILES string represents a valid molecule.
    FIXED: Now properly checks for heavy atoms (non-hydrogens) >= 3
    and rejects disconnected/separated molecules
    """
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return False
    
    smiles = smiles.strip()
    
    # FAST CHECK: Reject separated molecules (contains dots)
    if '.' in smiles:
        return False  # Disconnected components indicated by dots
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # CRITICAL FIX: Check heavy atoms (non-hydrogens), not total atoms
        heavy_atoms = mol.GetNumHeavyAtoms()  # This excludes hydrogens
        if heavy_atoms < 3:
            return False
            
        return True
    except Exception:
        return False

def passes_durrant_lab_filter(smiles: str) -> bool:
    """
    Apply Durrant's lab filter to remove improbable substructures.
    FIXED: More robust error handling, pattern checking, and disconnected molecule rejection.
    Returns True if molecule passes the filter (is acceptable), False otherwise.
    """
    if not smiles or not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return False
    
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return False
        
        # Check heavy atoms again (belt and suspenders approach)
        if mol.GetNumHeavyAtoms() < 3:
            return False
        
        # REJECT SEPARATED/DISCONNECTED MOLECULES (double check here too)
        fragments = Chem.rdmolops.GetMolFrags(mol, asMols=False)
        if len(fragments) > 1:
            return False  # Reject molecules with multiple disconnected parts
        
        # Define SMARTS patterns for problematic substructures
        problematic_patterns = [
            "C=[N-]",                    # Carbon double bonded to negative nitrogen
            "[N-]C=[N+]",               # Nitrogen anion bonded to nitrogen cation
            "[nH+]c[n-]",               # Aromatic nitrogen cation adjacent to nitrogen anion
            "[#7+]~[#7+]",              # Positive nitrogen connected to positive nitrogen
            "[#7-]~[#7-]",              # Negative nitrogen connected to negative nitrogen
            "[!#7]~[#7+]~[#7-]~[!#7]",  # Bridge: non-nitrogen - pos nitrogen - neg nitrogen - non-nitrogen
            "[#5]",                     # Boron atoms
            "O=[PH](=O)([#8])([#8])",   # Phosphoryl with hydroxyls
            "N=c1cc[#7]c[#7]1",         # Nitrogen in aromatic ring with another nitrogen
            "[$([NX2H1]),$([NX3H2])]=C[$([OH]),$([O-])]", # N=CH-OH or N=CH-O-
        ]
        
        # Check for metals (excluding common biologically relevant ions)
        metal_exclusions = {11, 12, 19, 20}  # Na, Mg, K, Ca
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            # More precise metal detection
            if atomic_num > 20 and atomic_num not in metal_exclusions:
                return False
        
        # Check for each problematic pattern
        for pattern in problematic_patterns:
            try:
                patt_mol = Chem.MolFromSmarts(pattern)
                if patt_mol is not None:
                    matches = mol.GetSubstructMatches(patt_mol)
                    if matches:
                        return False  # Found problematic substructure
            except Exception:
                # If SMARTS parsing fails, continue to next pattern
                continue
        
        return True  # Passed all checks
        
    except Exception:
        return False
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
    
    # Check validity first
    is_valid = (smiles is not None and 
                is_valid_smiles(smiles) and 
                passes_durrant_lab_filter(smiles))
    
    if is_valid:
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = None

    rewards = {
        "validity": 1.0 if is_valid else 0.0,
        "biological_diversity": compute_biological_diversity_score(mol),
        "charge_neutrality": compute_charge_neutrality_score(mol),
        "local_charge_penalty": compute_local_charge_penalty(mol),
        "lipinski": compute_enhanced_lipinski_reward(mol),
        "structural_complexity": compute_structural_complexity_reward(mol),
    }

    if not is_valid:
        # If not valid, set all chemistry-based rewards to 0
        for key in rewards:
            if key != "validity":
                rewards[key] = 0.0
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
    if smiles is None or not is_valid_smiles(smiles) or not passes_durrant_lab_filter(smiles):
        return 0.0
    mol = Chem.MolFromSmiles(smiles)
    return compute_lipinski_reward(mol)

# ========================
# PARETO-STYLE DYNAMIC REWARD CONTROLLER
# ========================

class ParetoRewardController:
    """
    Dynamic reward mixing based on Pareto optimality principles.
    Adapts reward weights based on current population performance.
    """
    
    def __init__(
        self,
        objectives: List[str] = None,
        history_size: int = 500,
        adaptation_rate: float = 0.1,
        min_weight: float = 0.05,
        max_weight: float = 0.95,
        pareto_pressure: float = 1.0,
        exploration_phase_length: int = 100
    ):
        """
        Args:
            objectives: List of objective names to track
            history_size: Size of rolling history for Pareto analysis
            adaptation_rate: How quickly weights adapt (0-1)
            min_weight: Minimum weight for any objective
            max_weight: Maximum weight for any objective  
            pareto_pressure: Higher = more aggressive toward Pareto front
            exploration_phase_length: Steps of pure exploration before adaptation
        """
        self.objectives = objectives or ["total", "sa", "validity", "diversity"]
        self.history_size = history_size
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.pareto_pressure = pareto_pressure
        self.exploration_phase_length = exploration_phase_length
        
        # Initialize weights equally
        n_objectives = len(self.objectives)
        self.weights = {obj: 1.0/n_objectives for obj in self.objectives}
        
        # History tracking
        self.objective_history = deque(maxlen=history_size)
        self.pareto_history = deque(maxlen=100)  # Track Pareto front evolution
        self.step_count = 0
        
        # Performance tracking
        self.objective_trends = {obj: deque(maxlen=50) for obj in self.objectives}
        self.stagnation_counters = {obj: 0 for obj in self.objectives}
        
    def update(self, batch_objectives: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update weights based on current batch performance.
        
        Args:
            batch_objectives: Dict of objective_name -> tensor of scores
            
        Returns:
            Updated weights dictionary
        """
        self.step_count += 1
        
        # Convert to numpy for easier manipulation
        batch_data = {}
        for obj_name, tensor_vals in batch_objectives.items():
            if obj_name in self.objectives:
                batch_data[obj_name] = tensor_vals.detach().cpu().numpy()
        
        # Store in history
        if len(batch_data) > 0:
            batch_size = len(batch_data[next(iter(batch_data))])
            for i in range(batch_size):
                point = {obj: batch_data[obj][i] for obj in self.objectives if obj in batch_data}
                if len(point) == len(self.objectives):  # Only store complete points
                    self.objective_history.append(point)
        
        # Skip adaptation during exploration phase
        if self.step_count <= self.exploration_phase_length:
            return self.weights.copy()
        
        # Compute current Pareto front
        current_front = self._compute_pareto_front()
        if len(current_front) > 0:
            self.pareto_history.append(len(current_front))
        
        # Adapt weights based on multiple criteria
        self._adapt_weights_pareto_driven(batch_data)
        self._adapt_weights_stagnation_driven(batch_data)
        self._adapt_weights_diversity_driven()
        
        # Ensure constraints
        self._normalize_weights()
        
        return self.weights.copy()
    
    def _compute_pareto_front(self) -> List[Dict[str, float]]:
        """Compute current Pareto front from history."""
        if len(self.objective_history) < 10:
            return []
        
        points = list(self.objective_history)
        pareto_front = []
        
        for i, point1 in enumerate(points):
            is_dominated = False
            for j, point2 in enumerate(points):
                if i != j and self._dominates(point2, point1):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(point1)
        
        return pareto_front
    
    def _dominates(self, point1: Dict[str, float], point2: Dict[str, float]) -> bool:
        """Check if point1 dominates point2 (higher is better for all objectives)."""
        better_in_all = True
        strictly_better_in_one = False
        
        for obj in self.objectives:
            if obj in point1 and obj in point2:
                if point1[obj] < point2[obj]:
                    better_in_all = False
                    break
                elif point1[obj] > point2[obj]:
                    strictly_better_in_one = True
        
        return better_in_all and strictly_better_in_one
    
    def _adapt_weights_pareto_driven(self, batch_data: Dict[str, np.ndarray]):
        """Adapt weights based on distance to Pareto front."""
        if len(self.objective_history) < 50:
            return
        
        pareto_front = self._compute_pareto_front()
        if len(pareto_front) == 0:
            return
        
        # Compute average distance to Pareto front for each objective
        obj_distances = {obj: [] for obj in self.objectives}
        
        for point in list(self.objective_history)[-100:]:  # Recent history
            min_distance = float('inf')
            closest_front_point = None
            
            for front_point in pareto_front:
                distance = sum((point[obj] - front_point[obj])**2 
                             for obj in self.objectives if obj in point and obj in front_point)
                if distance < min_distance:
                    min_distance = distance
                    closest_front_point = front_point
            
            if closest_front_point:
                for obj in self.objectives:
                    if obj in point and obj in closest_front_point:
                        obj_distances[obj].append(abs(point[obj] - closest_front_point[obj]))
        
        # Increase weight for objectives with larger gaps to Pareto front
        for obj in self.objectives:
            if obj_distances[obj]:
                avg_distance = np.mean(obj_distances[obj])
                # Higher distance = increase weight
                weight_adjustment = avg_distance * self.adaptation_rate * self.pareto_pressure
                self.weights[obj] = self.weights[obj] * (1 + weight_adjustment)
    
    def _adapt_weights_stagnation_driven(self, batch_data: Dict[str, np.ndarray]):
        """Increase weights for stagnating objectives."""
        for obj in self.objectives:
            if obj in batch_data:
                current_mean = np.mean(batch_data[obj])
                self.objective_trends[obj].append(current_mean)
                
                if len(self.objective_trends[obj]) >= 20:
                    recent_trend = np.array(list(self.objective_trends[obj])[-20:])
                    # Check for stagnation (low variance)
                    if np.std(recent_trend) < 0.01:  # Adjust threshold as needed
                        self.stagnation_counters[obj] += 1
                        # Boost weight for stagnating objectives
                        boost = min(0.1, self.stagnation_counters[obj] * 0.02)
                        self.weights[obj] += boost
                    else:
                        self.stagnation_counters[obj] = max(0, self.stagnation_counters[obj] - 1)
    
    def _adapt_weights_diversity_driven(self):
        """Adapt weights based on Pareto front diversity."""
        if len(self.pareto_history) < 10:
            return
        
        recent_front_sizes = list(self.pareto_history)[-10:]
        front_diversity = np.std(recent_front_sizes)
        
        # If diversity is low, boost exploration objectives
        if front_diversity < 1.0:  # Adjust threshold
            exploration_objectives = ["sa", "diversity"]  # Objectives that promote exploration
            for obj in exploration_objectives:
                if obj in self.weights:
                    self.weights[obj] += 0.05 * self.adaptation_rate
    
    def _normalize_weights(self):
        """Ensure weights are normalized and within bounds."""
        # Apply bounds
        for obj in self.weights:
            self.weights[obj] = np.clip(self.weights[obj], self.min_weight, self.max_weight)
        
        # Normalize to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            for obj in self.weights:
                self.weights[obj] /= total
        else:
            # Fallback to equal weights
            n = len(self.weights)
            for obj in self.weights:
                self.weights[obj] = 1.0 / n

    def get_mixed_reward(self, rewards_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute mixed reward using current weights.
        
        Args:
            rewards_dict: Dictionary of reward tensors
            
        Returns:
            Mixed reward tensor
        """
        mixed_reward = None
        
        for obj_name, weight in self.weights.items():
            if obj_name in rewards_dict:
                weighted_reward = weight * rewards_dict[obj_name]
                if mixed_reward is None:
                    mixed_reward = weighted_reward
                else:
                    mixed_reward += weighted_reward
        
        return mixed_reward if mixed_reward is not None else torch.zeros_like(list(rewards_dict.values())[0])
    
    def get_status(self) -> Dict[str, any]:
        """Get current status for logging."""
        pareto_front = self._compute_pareto_front()
        
        return {
            "weights": self.weights.copy(),
            "step_count": self.step_count,
            "pareto_front_size": len(pareto_front),
            "stagnation_counters": self.stagnation_counters.copy(),
            "history_size": len(self.objective_history),
            "avg_pareto_size": np.mean(list(self.pareto_history)) if self.pareto_history else 0
        }


# ========================
# RL TRAINING CONTROLLERS
# ========================

class AdaptiveKLController:
    """
    Adaptive KL controller with hard clipping and EMA smoothing.
    Prevents runaway beta values and exploding KL penalties.
    """

    def __init__(
        self,
        init_kl_coef: float = 0.2,
        target_kl: float = 6.0,
        horizon: int = 10000,
        max_kl_coef: float = 10.0,
        max_inc_factor: float = 2.0,
        ema_alpha: float = 0.9,
        kl_penalty_cap: float = 10.0,
    ):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon
        self.max_kl_coef = max_kl_coef
        self.max_inc_factor = max_inc_factor
        self.ema_alpha = ema_alpha
        self.kl_penalty_cap = kl_penalty_cap

        # Exponential moving average of KL
        self.ema_kl = None

    def update(self, current_kl: float, n_steps: int) -> None:
        # update EMA
        if self.ema_kl is None:
            self.ema_kl = current_kl
        else:
            self.ema_kl = (
                self.ema_alpha * self.ema_kl + (1 - self.ema_alpha) * current_kl
            )

        proportional_error = np.clip(
            (self.ema_kl - self.target) / self.target, -1.0, 1.0
        )
        mult = 1.0 + proportional_error * (n_steps / self.horizon)

        # cap growth
        if mult > self.max_inc_factor:
            mult = self.max_inc_factor

        # update beta
        new_val = self.value * mult
        self.value = min(new_val, self.max_kl_coef)

    def __call__(self) -> float:
        return self.value


def compute_kl_penalty(kl_vals: torch.Tensor, kl_coef: float, kl_penalty_cap: float):
    """
    Compute KL penalty with clipping.
    Returns (clipped_penalty, raw_penalty, kl_mean).
    """
    kl_mean = kl_vals.mean()
    raw_penalty = kl_coef * kl_mean
    clipped_penalty = torch.clamp(raw_penalty, max=kl_penalty_cap)
    return clipped_penalty, raw_penalty, kl_mean


class EnhancedEntropyController:
    def __init__(self, min_entropy: float = 0.5, max_entropy: float = 3.0,
                 target_entropy: float = 1.5):
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy
        self.target_entropy = target_entropy
        self.entropy_history: List[float] = []
        self.entropy_weight = 0.01

    def update_entropy_weight(self, current_entropy: float) -> float:
        self.entropy_history.append(float(current_entropy))
        if len(self.entropy_history) > 100:
            self.entropy_history = self.entropy_history[-100:]
        if len(self.entropy_history) >= 10:
            avg_entropy = np.mean(self.entropy_history[-10:])
            if avg_entropy < self.target_entropy * 0.8:
                self.entropy_weight = min(0.05, self.entropy_weight * 1.1)
            elif avg_entropy > self.target_entropy * 1.2:
                self.entropy_weight = max(0.001, self.entropy_weight * 0.95)
        return float(self.entropy_weight)

    def adjust_for_seq_len(self, seq_len: int, base_entropy: float = 1.5):
        seq_len = max(1, int(seq_len))
        self.target_entropy = float(base_entropy * np.log1p(seq_len) / np.log1p(10))
        self.target_entropy = float(np.clip(self.target_entropy, self.min_entropy, self.max_entropy))

    def reset(self):
        self.entropy_history.clear()
        self.entropy_weight = 0.01


class CurriculumManager:
    """Symmetric curriculum: 10→15→20→25→20→15→10→..."""
    def __init__(self, start_len: int = 10, max_len: int = 25,
                 step_increase: int = 5, steps_per_level: int = 30):
        self.start_len = start_len
        self.max_len = max_len
        self.step_increase = step_increase
        self.steps_per_level = steps_per_level
        self.current_max_len = start_len
        self.step_counter = 0
        self.direction = +1

    def get_max_new_tokens(self) -> int:
        return self.current_max_len

    def step(self) -> int:
        self.step_counter += 1
        if self.step_counter % self.steps_per_level == 0:
            if self.direction == +1:
                if self.current_max_len < self.max_len:
                    self.current_max_len += self.step_increase
                else:
                    self.direction = -1
                    self.current_max_len -= self.step_increase
            else:
                if self.current_max_len > self.start_len:
                    self.current_max_len -= self.step_increase
                else:
                    self.direction = +1
                    self.current_max_len += self.step_increase
            print(f"📈 Curriculum Update: max_new_tokens = {self.current_max_len}")
        return self.current_max_len

# ========================
# HELPERS
# ========================

def normalize_rewards(rewards: torch.Tensor, seq_len: int, mode: str = "sqrt") -> torch.Tensor:
    if seq_len <= 1 or mode == "none":
        return rewards
    if mode == "per_token":
        return rewards / float(seq_len)
    elif mode == "sqrt":
        return rewards / float(np.sqrt(seq_len))
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def reset_controllers_on_phase_change(prev_len: Optional[int], new_len: int,
                                      kl_controller: Optional[AdaptiveKLController] = None,
                                      entropy_controller: Optional[EnhancedEntropyController] = None,
                                      entropy_base: float = 1.5):
    if prev_len is None or prev_len == new_len:
        return
    if kl_controller is not None:
        kl_controller.reset()
    if entropy_controller is not None:
        entropy_controller.reset()
        entropy_controller.adjust_for_seq_len(new_len, base_entropy=entropy_base)


# ========================
# PPO LOSS
# ========================

def compute_ppo_loss(old_log_probs: torch.Tensor, new_log_probs: torch.Tensor,
                     rewards: torch.Tensor, clip_epsilon: float = 0.2,
                     baseline: Optional[torch.Tensor] = None,
                     seq_len: int = 1, reward_norm: str = "sqrt",
                     adv_clip: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    normed_rewards = normalize_rewards(rewards, seq_len, mode=reward_norm)
    if baseline is not None:
        advantage = normed_rewards - baseline.detach()
    else:
        advantage = normed_rewards
    if adv_clip is not None:
        advantage = torch.clamp(advantage, -float(adv_clip), float(adv_clip))
    else:
        default_clip = 2.0 * np.sqrt(max(1, seq_len))
        advantage = torch.clamp(advantage, -default_clip, default_clip)
    log_ratio = torch.clamp(new_log_probs - old_log_probs, -10.0, 10.0)
    ratio = torch.exp(log_ratio)
    adv_expanded = advantage.unsqueeze(1) if advantage.dim() == 1 else advantage
    surr1 = ratio * adv_expanded
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv_expanded
    ppo_loss = -torch.min(surr1, surr2).sum(dim=1).mean()
    return ppo_loss, advantage.detach()


def compute_kl_divergence(old_action_probs: torch.Tensor, new_action_probs: torch.Tensor) -> torch.Tensor:
    old_probs = old_action_probs.clamp_min(1e-12)
    new_probs = new_action_probs.clamp_min(1e-12)
    kl_per_step = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(dim=-1)
    return kl_per_step.sum(dim=1)


def compute_entropy_bonus(action_probs: torch.Tensor) -> torch.Tensor:
    probs = action_probs.clamp_min(1e-12)
    entropy_per_step = -(probs * torch.log(probs)).sum(dim=-1)
    return entropy_per_step.sum(dim=1)

# ========================
# BATCH REWARD COMPUTATION
# ========================

def batch_compute_rewards_pareto(
    selfies_list: List[str],
    reward_mode: str = "mix",
    reward_mix: float = 0.5,
    pareto_controller: Optional[ParetoRewardController] = None
) -> Dict[str, torch.Tensor]:
    """
    Drop-in replacement for batch_compute_rewards with Pareto support.
    
    Args:
        selfies_list: List of SELFIES strings  
        reward_mode: "chemq3", "sa", "mix", or "pareto" 
        reward_mix: Weight for comprehensive rewards when mixing (0-1)
        pareto_controller: ParetoRewardController instance for "pareto" mode
        
    Returns:
        Dictionary containing reward tensors (same format as original)
    """
    batch_size = len(selfies_list)
    
    validity_vals = []
    lipinski_vals = []
    total_rewards = []
    sa_rewards = []

    # Compute all individual rewards
    for selfies_str in selfies_list:
        smiles = selfies_to_smiles(selfies_str)
        
        # Check validity comprehensively  
        is_valid = (smiles is not None and 
                   is_valid_smiles(smiles) and 
                   passes_durrant_lab_filter(smiles))
        
        if reward_mode in ["chemq3", "mix", "pareto"]:
            r = compute_comprehensive_reward(selfies_str)
            validity_vals.append(r.get('validity', 0.0))
            lipinski_vals.append(r.get('lipinski', 0.0))

        if reward_mode in ["sa", "mix", "pareto"]:
            sa = compute_sa_reward(selfies_str) if is_valid else 0.0
            sa_rewards.append(sa)

        # Store individual comprehensive reward for pareto mode
        if reward_mode in ["chemq3", "pareto"]:
            total_rewards.append(r.get('total', 0.0))
        elif reward_mode == "sa":
            total_rewards.append(sa)
        elif reward_mode == "mix":
            r_total = r.get("total", 0.0) if 'r' in locals() else 0.0
            sa_val = sa if 'sa' in locals() else 0.0
            mixed = reward_mix * r_total + (1.0 - reward_mix) * sa_val
            total_rewards.append(mixed)

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
    
    # Compute diversity reward
    valid_smiles = []
    for selfies_str in selfies_list:
        smiles = selfies_to_smiles(selfies_str)
        if smiles and is_valid_smiles(smiles) and passes_durrant_lab_filter(smiles):
            valid_smiles.append(smiles)
    
    diversity_score = len(set(valid_smiles)) / max(1, len(valid_smiles))
    result["diversity_rewards"] = torch.full((batch_size,), diversity_score, dtype=torch.float32)
    
    # Apply Pareto mixing if requested
    if reward_mode == "pareto" and pareto_controller is not None:
        # Prepare objectives for controller update
        batch_objectives = {
            "total": result["total_rewards"],
            "validity": result.get("validity_rewards", torch.zeros(batch_size)),
            "diversity": result["diversity_rewards"]
        }
        
        if "sa_rewards" in result:
            batch_objectives["sa"] = result["sa_rewards"]
        
        # Update controller and get new weights
        updated_weights = pareto_controller.update(batch_objectives)
        
        # Compute mixed reward using adaptive weights
        mixed_reward = pareto_controller.get_mixed_reward(batch_objectives)
        result["total_rewards"] = mixed_reward
        
        # Store weights for logging
        result["pareto_weights"] = updated_weights

    return result

# Legacy
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
        smiles = selfies_to_smiles(selfies_str)
        
        # Check validity comprehensively
        is_valid = (smiles is not None and 
                   is_valid_smiles(smiles) and 
                   passes_durrant_lab_filter(smiles))
        
        if reward_mode == "chemq3":
            r = compute_comprehensive_reward(selfies_str)
            validity_vals.append(r.get('validity', 0.0))
            lipinski_vals.append(r.get('lipinski', 0.0))
            total_rewards.append(r.get('total', 0.0))

        elif reward_mode == "sa":
            sa = compute_sa_reward(selfies_str) if is_valid else 0.0
            sa_rewards.append(sa)
            total_rewards.append(sa)

        elif reward_mode == "mix":
            r = compute_comprehensive_reward(selfies_str)
            sa = compute_sa_reward(selfies_str) if is_valid else 0.0
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
        if smiles and is_valid_smiles(smiles) and passes_durrant_lab_filter(smiles):
            valid_smiles.append(smiles)
    
    metrics["num_valid"] = len(valid_smiles)
    metrics["num_unique"] = len(set(valid_smiles))
    metrics["diversity_ratio"] = len(set(valid_smiles)) / max(1, len(valid_smiles))
    
    # Add loss components
    metrics.update(loss_dict)
    

    return metrics