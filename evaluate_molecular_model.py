# evaluate_molecular_model.py
import os
import sys
import json
import argparse
import random
from typing import List, Optional
from tqdm import tqdm

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import selfies as sf
import pandas as pd

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Add local path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from FastChemTokenizerHF import FastChemTokenizerSelfies
from ChemQ3MTP import ChemQ3MTPForCausalLM

# ----------------------------
# Robust Conversion & Validation (as per your spec)
# ----------------------------

def selfies_to_smiles(selfies_str: str) -> Optional[str]:
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
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol is not None

def get_sa_label_and_confidence(selfies_str: str) -> tuple[str, float]:
    """Get SA label (Easy/Hard) and confidence from the model's SA classifier."""
    try:
        from ChemQ3MTP.rl_utils import get_sa_classifier
        classifier = get_sa_classifier()
        if classifier is None:
            return "Unknown", 0.0
        
        # Get raw classifier output: [{'label': 'Easy', 'score': 0.9187200665473938}]
        result = classifier(selfies_str, truncation=True, max_length=128)[0]
        return result["label"], result["score"]
    except Exception as e:
        return "Unknown", 0.0

def get_morgan_fingerprint_from_smiles(smiles: str, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

def tanimoto_sim(fp1, fp2):
    from rdkit.DataStructs import TanimotoSimilarity
    return TanimotoSimilarity(fp1, fp2)

# ----------------------------
# Main Evaluation Function
# ----------------------------

def evaluate_model(
    model_path: str,
    train_data_path: str = "../data/chunk_5.csv",
    n_samples: int = 1000,
    seed: int = 42,
    max_gen_len: int = 32
):
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Evaluating model at: {model_path}")
    print(f"   Device: {device} | Samples: {n_samples} | Seed: {seed}\n")

    # Load tokenizer and model
    tokenizer = FastChemTokenizerSelfies.from_pretrained("../selftok_core")
    model = ChemQ3MTPForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load training set and normalize SELFIES (remove spaces)
    print("📂 Loading and normalizing training set for novelty...")
    train_df = pd.read_csv(train_data_path)
    train_selfies_clean = set()
    for s in train_df["SELFIES"].dropna().astype(str):
        clean_s = s.replace(" ", "")
        train_selfies_clean.add(clean_s)
    print(f"   Training set size: {len(train_selfies_clean)} unique (space-free) SELFIES\n")

    # === MTP-AWARE GENERATION ===
    print("GenerationStrategy: Using MTP-aware generation...")
    all_selfies_raw = []
    batch_size = 32
    num_batches = (n_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating"):
            current_batch_size = min(batch_size, n_samples - len(all_selfies_raw))
            if current_batch_size <= 0:
                break

            input_ids = torch.full(
                (current_batch_size, 1),
                tokenizer.bos_token_id,
                dtype=torch.long,
                device=device
            )

            if hasattr(model, 'generate_with_logprobs'):
                try:
                    outputs = model.generate_with_logprobs(
                        input_ids=input_ids,
                        max_new_tokens=25,
                        temperature=1.0,
                        top_k=50,
                        top_p=0.95,
                        do_sample=True,
                        return_probs=True,
                        tokenizer=tokenizer
                    )
                    batch_selfies = outputs[0]  # list of raw SELFIES (may have spaces)
                except Exception as e:
                    print(f"⚠️ MTP generation failed: {e}. Falling back.")
                    gen_tokens = model.generate(
                        input_ids,
                        max_length=max_gen_len,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=1.0,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    batch_selfies = [
                        tokenizer.decode(seq, skip_special_tokens=True)
                        for seq in gen_tokens
                    ]
            else:
                gen_tokens = model.generate(
                    input_ids,
                    max_length=max_gen_len,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                batch_selfies = [
                    tokenizer.decode(seq, skip_special_tokens=True)
                    for seq in gen_tokens
                ]

            all_selfies_raw.extend(batch_selfies)
            if len(all_selfies_raw) >= n_samples:
                break

    all_selfies_raw = all_selfies_raw[:n_samples]
    print(f"\n✅ Generated {len(all_selfies_raw)} raw SELFIES strings.\n")

    # Process: SELFIES → clean SELFIES → SMILES → valid molecules
    valid_records = []
    print("🧪 Processing SELFIES and converting to SMILES...")
    for i, raw_selfies in enumerate(tqdm(all_selfies_raw, desc="Converting")):
        # Clean the SELFIES (remove spaces as tokenizer uses whitespace)
        clean_selfies = raw_selfies.replace(" ", "")
        
        # Convert to SMILES
        smiles = selfies_to_smiles(clean_selfies)
        
        if smiles and is_valid_smiles(smiles):
            valid_records.append({
                "raw_selfies": raw_selfies,
                "selfies_clean": clean_selfies,
                "selfies": clean_selfies,  # canonical version
                "smiles": smiles.strip()
            })

    # >>> DEBUG: Print multiple examples and SA label analysis <<<
    if valid_records:
        print("\n🔍 DEBUG: Sample generated molecules")
        print("-" * 70)
        for i in range(min(5, len(valid_records))):
            example = valid_records[i]
            print(f"Example {i+1}:")
            print(f"  Raw SELFIES : {example['raw_selfies'][:80]}{'...' if len(example['raw_selfies']) > 80 else ''}")
            print(f"  SMILES      : {example['smiles']}")
            
            # Get SA label and confidence
            label, confidence = get_sa_label_and_confidence(example['raw_selfies'])
            print(f"  SA Label    : {label} (confidence: {confidence:.3f})")
            
            if i == 0:
                # Test SA classifier with simple molecules
                simple_label, simple_conf = get_sa_label_and_confidence('[C]')
                benzene_label, benzene_conf = get_sa_label_and_confidence('[c] [c] [c] [c] [c] [c] [Ring1] [=Branch1]')
                print(f"  🧪 SA Test - Simple molecule: {simple_label} ({simple_conf:.3f})")
                print(f"  🧪 SA Test - Benzene: {benzene_label} ({benzene_conf:.3f})")
            
            # Check molecule properties
            mol = Chem.MolFromSmiles(example['smiles'])
            if mol:
                print(f"  Atoms       : {mol.GetNumAtoms()}")
                print(f"  Bonds       : {mol.GetNumBonds()}")
            print()
        print("-" * 70)
        
        # SA Label distribution analysis
        sa_labels = []
        for r in valid_records[:100]:
            label, _ = get_sa_label_and_confidence(r["raw_selfies"])
            sa_labels.append(label)
        
        easy_count = sa_labels.count("Easy")
        hard_count = sa_labels.count("Hard")
        unknown_count = sa_labels.count("Unknown")
        
        print(f"🔍 SA Label Analysis (first 100 molecules):")
        print(f"  Easy to synthesize: {easy_count}/100 ({easy_count}%)")
        print(f"  Hard to synthesize: {hard_count}/100 ({hard_count}%)")
        if unknown_count > 0:
            print(f"  Unknown/Failed: {unknown_count}/100 ({unknown_count}%)")
    else:
        print("\n⚠️  WARNING: No valid molecules generated in sample!")
    # <<< END DEBUG >>>

    # Now compute metrics...
    validity = len(valid_records) / n_samples
    
    unique_valid = list({r["selfies_clean"]: r for r in valid_records}.values())
    uniqueness = len(unique_valid) / len(valid_records) if valid_records else 0.0

    novel_count = sum(1 for r in unique_valid if r["selfies_clean"] not in train_selfies_clean)
    novelty = novel_count / len(unique_valid) if unique_valid else 0.0

    # SA Label Counts (using model's SA classifier)
    sa_labels_all = []
    for r in unique_valid:
        label, _ = get_sa_label_and_confidence(r["raw_selfies"])
        sa_labels_all.append(label)
    
    easy_total = sa_labels_all.count("Easy")
    hard_total = sa_labels_all.count("Hard")
    unknown_total = sa_labels_all.count("Unknown")
    total_labeled = len(sa_labels_all)

    # Internal Diversity (on SMILES)
    if len(unique_valid) >= 2:
        fps = []
        for r in unique_valid:
            fp = get_morgan_fingerprint_from_smiles(r["smiles"])
            if fp is not None:
                fps.append(fp)
        if len(fps) >= 2:
            total_sim, count = 0.0, 0
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    total_sim += tanimoto_sim(fps[i], fps[j])
                    count += 1
            internal_diversity = 1.0 - (total_sim / count)
        else:
            internal_diversity = 0.0
    else:
        internal_diversity = 0.0

    # ----------------------------
    # Final Summary
    # ----------------------------
    print("\n" + "="*55)
    print("📊 MOLECULAR GENERATION EVALUATION SUMMARY")
    print("="*55)
    print(f"Model Path       : {model_path}")
    print(f"Generation Mode  : {'MTP-aware' if hasattr(model, 'generate_with_logprobs') else 'Standard'}")
    print(f"Samples Generated: {n_samples}")
    print("-"*55)
    print(f"Validity         : {validity:.4f} ({len(valid_records)}/{n_samples})")
    print(f"Uniqueness       : {uniqueness:.4f} (unique valid)")
    print(f"Novelty (vs train): {novelty:.4f} (space-free SELFIES)")
    print(f"Synthesis Labels : Easy: {easy_total}/{total_labeled} ({easy_total/max(1,total_labeled)*100:.1f}%) | Hard: {hard_total}/{total_labeled} ({hard_total/max(1,total_labeled)*100:.1f}%)")
    if unknown_total > 0:
        print(f"                   Unknown: {unknown_total}/{total_labeled} ({unknown_total/max(1,total_labeled)*100:.1f}%)")
    print(f"Internal Diversity: {internal_diversity:.4f} (1 - avg Tanimoto)")
    print("="*55)

    results = {
        "model_path": model_path,
        "generation_mode": "MTP-aware" if hasattr(model, 'generate_with_logprobs') else "standard",
        "n_samples": n_samples,
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "sa_easy_count": easy_total,
        "sa_hard_count": hard_total,
        "sa_easy_percentage": easy_total/max(1,total_labeled)*100,
        "sa_hard_percentage": hard_total/max(1,total_labeled)*100,
        "internal_diversity": internal_diversity,
        "valid_molecules_count": len(valid_records)
    }
    
    if unknown_total > 0:
        results["sa_unknown_count"] = unknown_total
        results["sa_unknown_percentage"] = unknown_total/max(1,total_labeled)*100

    output_json = os.path.join(model_path, "evaluation_summary.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_json}")

    return results

# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate molecular generative model with MTP-aware generation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of molecules to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_data", type=str, default="../data/chunk_5.csv", help="Training data CSV")

    args = parser.parse_args()
    evaluate_model(
        model_path=args.model_path,
        train_data_path=args.train_data,
        n_samples=args.n_samples,
        seed=args.seed
    )