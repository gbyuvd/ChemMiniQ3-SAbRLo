#!/usr/bin/env python3
# Refactored PPO-KL training script using ChemQ3MTP module with Pareto Controller

# rl_finetuning.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import os
import torch
import numpy as np
from tqdm import tqdm
from FastChemTokenizerHF import FastChemTokenizerSelfies
from ChemQ3MTP import ChemQ3MTPForCausalLM 
from ChemQ3MTP.rl_utils import (
    CurriculumManager, 
    AdaptiveKLController, 
    batch_compute_rewards_pareto,  # NEW: Import the Pareto version
    ParetoRewardController,        # NEW: Import Pareto controller
    compute_ppo_loss, 
    compute_kl_divergence, 
    compute_entropy_bonus, 
    compute_kl_penalty
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # --- Load tokenizer ---
    tokenizer = FastChemTokenizerSelfies.from_pretrained("../selftok_core")

    # --- Load model ---
    model = ChemQ3MTPForCausalLM.from_pretrained("./chunk-4")  # Updated to use new model class
    model.tokenizer = tokenizer
    model.to(device)

    # --- RL fine-tuning setup ---
    print("\n🎯 Phase 2: RL Fine-tuning with PPO + Curriculum Learning + Pareto Adaptation")
    model.set_mtp_training(False)
    
    # Initialize KL controller - Using correct parameter name based on class definition
    kl_controller = AdaptiveKLController(
        init_kl_coef=0.1,
        target_kl=0.01,
        horizon=100,        # <-- use horizon instead of kl_horizon
        max_kl_coef=100.0,  # optional
        ema_alpha=0.9,      # optional
        kl_penalty_cap=10.0 # optional
    )
    model.kl_controller = kl_controller  # Set on model for consistency
    
    # NEW: Initialize Pareto controller for dynamic reward mixing
    pareto_controller = ParetoRewardController(
        objectives=["total", "sa", "validity", "diversity"],
        history_size=1000,              # Larger history for better adaptation
        adaptation_rate=0.05,           # Conservative adaptation
        min_weight=0.05,               # Minimum 5% weight for any objective
        max_weight=0.7,                # Maximum 70% weight for any objective
        pareto_pressure=0.8,           # Moderate pressure toward Pareto front
        exploration_phase_length=200   # 200 steps of exploration before adaptation
    )
    print(f"🎛️ Initialized Pareto controller with objectives: {pareto_controller.objectives}")
    print(f"Initial weights: {pareto_controller.weights}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    curriculum = CurriculumManager(start_len=10, max_len=25, step_increase=5, steps_per_level=100)
    baseline = None
    gamma = 0.95

    # Dummy input (BOS-only batch)
    batch_size = 4
    dummy_input = tokenizer([tokenizer.bos_token] * batch_size, return_tensors="pt", padding=True)
    input_ids = dummy_input.input_ids.to(device)

    # Training config
    total_steps = 4500
    checkpoint_steps = {total_steps // 4, total_steps // 2, 3 * total_steps // 4, total_steps}
    checkpoint_dir = "./ppo_checkpoints_pareto"  # NEW: Different checkpoint dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- RL Training Loop with tqdm ---
    for step in tqdm(range(total_steps), desc="RL Training with Pareto"):
        global_step = step  # Define global_step for KL controller
        max_new_tokens = curriculum.get_max_new_tokens()

        # === PPO Rollout ===
        with torch.no_grad():
            selfies_list, old_log_probs, _, old_action_probs = model.generate_with_logprobs(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                return_probs=True
            )
            old_log_probs = old_log_probs.detach()
            old_action_probs = old_action_probs.detach()

        # === Generate new rollouts for PPO update ===
        selfies_list, new_log_probs, token_ids, new_action_probs = model.generate_with_logprobs(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            return_probs=True,
            tokenizer=tokenizer,
        )

        # === Compute rewards using NEW Pareto-aware rl_utils ===
        rewards_dict = batch_compute_rewards_pareto(
            selfies_list=selfies_list,
            reward_mode="pareto",  # NEW: Use Pareto adaptive mixing
            pareto_controller=pareto_controller
        )
        rewards = rewards_dict["total_rewards"].to(device)

        # === Compute PPO loss ===
        ppo_loss, advantage = compute_ppo_loss(
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            rewards=rewards,
            clip_epsilon=0.2,
            baseline=baseline
        )

        # === Compute KL divergence and update controller ===
        # Compute KL divergence per batch
        # === Compute KL divergence and update controller ===
        kl_div = compute_kl_divergence(old_action_probs, new_action_probs)
        kl_mean = kl_div.mean().item()

        # Update KL controller using EMA-smoothed KL
        kl_controller.update(kl_mean, n_steps=global_step)
        beta = kl_controller()  # get current coefficient

        # Compute clipped KL penalty
        kl_penalty, raw_kl_penalty, kl_mean_tensor = compute_kl_penalty(
            kl_div, beta, kl_controller.kl_penalty_cap
        )

        # --- Logging (safe, interpretable values) ---
        logs = {}
        logs["kl_mean"] = kl_mean_tensor.item()
        logs["kl_beta"] = beta
        logs["kl_penalty_raw"] = raw_kl_penalty.item()
        logs["kl_penalty_clipped"] = kl_penalty.item()

        # === Compute entropy bonus with adaptive weighting ===
        entropy_per_example = compute_entropy_bonus(new_action_probs)
        entropy = entropy_per_example.mean()

        # Update entropy controller
        adaptive_entropy_weight = model.entropy_controller.update_entropy_weight(entropy.item())
        entropy_bonus = adaptive_entropy_weight * entropy

        # === Total loss ===
        total_policy_loss = ppo_loss + kl_penalty
        total_loss = total_policy_loss - entropy_bonus

        # Optional regularization
        reg_loss = 1e-7 * sum(p.pow(2).sum() for p in model.parameters())
        total_loss = total_loss + reg_loss

        # === Backward pass ===
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # === Update baseline ===
        reward_tensor = rewards.mean()
        baseline = reward_tensor if baseline is None else gamma * baseline + (1 - gamma) * reward_tensor

        # Curriculum update
        curriculum.step()

        # Checkpointing with Pareto controller state
        if (step + 1) in checkpoint_steps:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{step+1}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            
            # NEW: Save Pareto controller state
            pareto_state = pareto_controller.get_status()
            
            torch.save({
                'step': step + 1,
                'optimizer_state_dict': optimizer.state_dict(),
                'baseline': baseline.item(),
                'curriculum_state': {
                    'current_max_len': curriculum.current_max_len,
                    'step_counter': curriculum.step_counter
                },
                # NEW: Save Pareto controller state
                'pareto_state': {
                    'weights': pareto_state['weights'],
                    'step_count': pareto_state['step_count'],
                    'stagnation_counters': pareto_state['stagnation_counters']
                }
            }, os.path.join(checkpoint_path, 'training_state.pt'))
            print(f"\n💾 Checkpoint saved at step {step+1} -> {checkpoint_path}")
            print(f"   Pareto weights: {pareto_state['weights']}")

        # Enhanced logging every 50 steps
        if step % 50 == 0:
            # Compute validity rate for logging
            validity_count = 0
            for selfies in selfies_list[:10]:  # Check first 10 samples like original
                from ChemQ3MTP.rl_utils import selfies_to_smiles
                smiles = selfies_to_smiles(selfies)
                if smiles and smiles != "":
                    validity_count += 1
            validity_rate = validity_count / max(1, min(10, len(selfies_list)))
            
            # Compute lipinski score for logging (even though not used in rewards)
            lipinski_scores = []
            for selfies in selfies_list[:10]:
                from rdkit import Chem
                smiles = selfies_to_smiles(selfies)
                mol = Chem.MolFromSmiles(smiles) if smiles else None
                if mol:
                    from rdkit.Chem import Lipinski, Descriptors
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    hbd = Lipinski.NumHDonors(mol)
                    hba = Lipinski.NumHAcceptors(mol)
                    rules = [250 < mw <= 500, logp <= 5, hbd <= 5, hba <= 10]
                    lipinski_score = sum(rules) / 4.0
                    lipinski_scores.append(lipinski_score)
            lipinski_score = np.mean(lipinski_scores) if lipinski_scores else 0.0

            # Extract individual reward components
            avg_total_reward = rewards_dict["total_rewards"].mean().item()
            avg_sa_reward = rewards_dict.get("sa_rewards", torch.zeros(1)).mean().item()
            avg_validity = rewards_dict.get("validity_rewards", torch.zeros(1)).mean().item()
            avg_diversity = rewards_dict.get("diversity_rewards", torch.zeros(1)).mean().item()
            
            # NEW: Get Pareto controller status
            pareto_status = pareto_controller.get_status()
            current_weights = pareto_status['weights']
            pareto_front_size = pareto_status['pareto_front_size']

            log_line = (
                f"\n[RL Step {step}] "
                f"Loss={total_loss.item():.4f} | "
                f"Valid={validity_rate:.3f} | "
                f"Lipinski={lipinski_score:.3f} | "
                f"Reward={avg_total_reward:.3f} | "
                f"SA={avg_sa_reward:.3f} | "
                f"Diversity={avg_diversity:.3f} | "
                f"Entropy={entropy.item():.3f} | "
                f"EntropyW={adaptive_entropy_weight:.4f} | "
                f"KL_Beta={beta:.4f} | "
                f"KL_Mean={kl_mean:.4f}"
            )
            print(log_line)
            
            # NEW: Enhanced Pareto logging
            print(f"🎛️ Pareto Status:")
            print(f"   Weights: Total={current_weights.get('total', 0):.3f}, SA={current_weights.get('sa', 0):.3f}, "
                  f"Valid={current_weights.get('validity', 0):.3f}, Div={current_weights.get('diversity', 0):.3f}")
            print(f"   Front size: {pareto_front_size}, History: {pareto_status['history_size']}")
            
            # Show weight changes if past exploration phase
            if step > pareto_controller.exploration_phase_length:
                stag_counters = pareto_status['stagnation_counters']
                if any(v > 0 for v in stag_counters.values()):
                    print(f"   Stagnation: {stag_counters}")

            # Sample conversion for display
            sample_selfies = selfies_list[0][:100]
            sample_smiles = selfies_to_smiles(selfies_list[0]) or "Invalid"
            print(f"  Sample SELFIES: {sample_selfies}")
            print(f"  Sample SMILES: {sample_smiles}")

        # NEW: Special logging every 200 steps for Pareto analysis
        if step % 200 == 0 and step > 0:
            pareto_status = pareto_controller.get_status()
            print(f"\n📊 Pareto Analysis at Step {step}:")
            print(f"   Average Pareto front size: {pareto_status['avg_pareto_size']:.1f}")
            print(f"   Adaptation active: {'Yes' if step > pareto_controller.exploration_phase_length else 'No'}")
            
            # Show recent objective performance
            recent_rewards = {
                'Total': avg_total_reward,
                'SA': avg_sa_reward,
                'Validity': avg_validity,
                'Diversity': avg_diversity
            }
            print(f"   Current objectives: {recent_rewards}")

    print("🎉 Training complete with Pareto adaptation!")
    
    # NEW: Final Pareto summary
    final_status = pareto_controller.get_status()
    print(f"\n🏁 Final Pareto Controller Summary:")
    print(f"   Final weights: {final_status['weights']}")
    print(f"   Total adaptations: {final_status['step_count']}")
    print(f"   Final Pareto front size: {final_status['pareto_front_size']}")

if __name__ == "__main__":
    main()