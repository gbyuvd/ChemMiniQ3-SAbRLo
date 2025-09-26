#!/usr/bin/env python3
# Refactored PPO-KL training script using ChemQ3MTP module

# rl_finetuning.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import os
import torch
from tqdm import tqdm
from FastChemTokenizerHF import FastChemTokenizerSelfies
from ChemQ3MTP import ChemQ3MTPForCausalLM, CurriculumManager
from ChemQ3MTP.rl_utils import AdaptiveKLController, batch_compute_rewards, compute_ppo_loss, compute_kl_divergence, compute_entropy_bonus

# ... rest of your RL script

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # --- Load tokenizer ---
    tokenizer = FastChemTokenizerSelfies.from_pretrained("../selftok_core")

    # --- Load model ---
    model = ChemQ3MTPForCausalLM.from_pretrained("./CMQ3-o-1")  # Updated to use new model class
    model.tokenizer = tokenizer
    model.to(device)

    # --- RL fine-tuning setup ---
    print("\n🎯 Phase 2: RL Fine-tuning with PPO + Curriculum Learning")
    model.set_mtp_training(False)
    
    # Initialize KL controller
    kl_controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=0.01, kl_horizon=100)
    model.kl_controller = kl_controller  # Set on model for consistency
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    curriculum = CurriculumManager(start_len=10, max_len=25, step_increase=5, steps_per_level=100)
    baseline = None
    gamma = 0.95

    # Dummy input (BOS-only batch)
    batch_size = 4
    dummy_input = tokenizer([tokenizer.bos_token] * batch_size, return_tensors="pt", padding=True)
    input_ids = dummy_input.input_ids.to(device)

    # Training config
    total_steps = 10000
    checkpoint_steps = {total_steps // 4, total_steps // 2, 3 * total_steps // 4, total_steps}
    checkpoint_dir = "./ppo_checkpoints_test"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- RL Training Loop with tqdm ---
    for step in tqdm(range(total_steps), desc="RL Training"):
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

        # === Compute rewards using rl_utils ===
        rewards_dict = batch_compute_rewards(
            selfies_list=selfies_list,
            reward_mode="sa",  # SA-only mode
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
        kl_div = compute_kl_divergence(old_action_probs, new_action_probs)
        beta = kl_controller.update(kl_div.mean().item())
        kl_penalty = beta * kl_div.mean()

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

        # Checkpointing
        if (step + 1) in checkpoint_steps:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{step+1}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            torch.save({
                'step': step + 1,
                'optimizer_state_dict': optimizer.state_dict(),
                'baseline': baseline.item(),
                'curriculum_state': {
                    'current_max_len': curriculum.current_max_len,
                    'step_counter': curriculum.step_counter
                }
            }, os.path.join(checkpoint_path, 'training_state.pt'))
            print(f"\n💾 Checkpoint saved at step {step+1} -> {checkpoint_path}")

        # Logging every 50 steps
        if step % 50 == 0:
            log_line = (
                f"\n[RL Step {step}] "
                f"Loss={total_loss.item():.4f} | "
                f"PPO Loss={ppo_loss.item():.4f} | "
                f"KL Penalty={kl_penalty.item():.4f} | "
                f"KL Coef={beta:.4f} | "
                f"Entropy={entropy.item():.3f} | "
                f"EntropyW={adaptive_entropy_weight:.4f} | "
                f"Avg Reward={rewards.mean().item():.3f}"
            )
            print(log_line)

            # Compute validity rate for logging
            validity_count = 0
            for selfies in selfies_list[:5]:  # Check first 5 samples
                from ChemQ3MTP.rl_utils import selfies_to_smiles
                smiles = selfies_to_smiles(selfies)
                if smiles and smiles != "":
                    validity_count += 1
            validity_rate = validity_count / min(5, len(selfies_list))
            
            print(f"  Sample Validity Rate: {validity_rate:.3f}")
            print(f"  Sample SELFIES: {selfies_list[0][:100]}")

    print("🎉 Training complete!")

if __name__ == "__main__":
    main()