
# 🧬 ChemMiniQ3-SAbRLo (Synthetic Accessibility with Bioaware RL — Optimized)

ChemMiniQ3-SAbRLo is a lightweight experimental generative model for chemistry, built on mini **Qwen3**, designed for **rapid prototyping of HuggingFace `AutoModel` and `AutoTokenizer` compatibility**, and **fast iteration of Multi-Token Prediction (MTP) and RL fine-tuning algorithms/rewards**.

It introduces a **new reinforcement learning framework** as the next iteration of [ChemMiniQ3-HoriFIE](https://huggingface.co/gbyuvd/ChemMiniQ3-HoriFIE), combining:

- 🧩 **Synthetic Accessibility (SA) Rewards** — guiding generation with a classifier (`gbyuvd/synthaccess-chemselfies`) to favor molecules that are easier to synthesize.  
- 🔄 **Cyclical Gradual Generation** — a curriculum learning strategy that **gradually increases molecule length up to 25 tokens**, then **resets and repeats**, enabling faster RL convergence and stable prototyping.

*Prototype research code — not production-ready. Built for speed, not scale - yet.*

Example of a generated molecule, found no identical mol in PubChem

`O=C(O)CC=1CCCCC=1C2=CC=CC(=C2)NC(=O)CC3=CC=CC=C3CCC`

![image](https://cdn-uploads.huggingface.co/production/uploads/667da868d653c0b02d6a2399/-etV70JXkUe1G7Sy1n12z.png)

---

## ⚙️ Core Features

- ✅ **Qwen3 Mini Backbone** – Efficient causal LM architecture, compatible with `transformers.AutoModelForCausalLM`  
- ✅ **Multi-Token Prediction (MTP Head)** – Parallel prediction of 1–3 future tokens, implemented as a plug-and-play head compatible with `AutoModel`  
- ✅ **Horizon Loss** – Weighted multi-horizon objectives for long-term coherence  
- ✅ **SELFIES-native Tokenizer** – Robust encoding with [FastChemTokenizer](https://github.com/gbyuvd/FastChemTokenizer)  
- ✅ **Ranger21 Optimizer** – Warmup/warmdown scheduling for stable training  
- ✅ **Gradient Checkpointing & Streaming Dataset Loader** – Lightweight, hardware-friendly, optimized for rapid RL prototyping  

---
# 🧪 Reinforcement Learning Enhancements

## 1️⃣ SA-Guided PPO-KL Fine-Tuning
- Uses `gbyuvd/synthaccess-chemselfies` as a **reward model**  
- Rewards molecules predicted as **"Easy"** to synthesize  
- Penalizes molecules predicted as **"Hard"**  
- Designed for **rapid reward ablation**: SA-only, ChemQ3-only, or mixed modes  
- Tries to be compatible with HuggingFace `Trainer` and `PPOTrainer` for easy RL experimentation  

## 2️⃣ Symmetric Curriculum with Normalized Rewards
- Generation length increases and decreases smoothly: **10 → 15 → 20 → 25 → 20 → 15 → 10 …**  
- Avoids sharp resets by cycling symmetrically instead of jumping from max back to min  
- **[Previous cyclical approach - now enhanced]**: Gradually increases max generation length, but now uses **symmetric cycling** to avoid sharp transitions  
- Rewards are normalized by sequence length (default: √len) to stabilize training across different rollout sizes  
- KL and entropy controllers are reset and recalibrated at each curriculum phase change  
- Entropy targets scale with sequence length, encouraging consistent exploration at both short and long contexts  
- Why **25**? Because **faster RL training requires shorter sequences** to enable rapid iteration — 25 tokens potentially could strike the optimal balance between structural complexity and training speed, allowing 2–3x more gradient steps per epoch compared to 30+ token sequences  

> 💡 *Note: The average SELFIES sequence length in our ~3M dataset is 33.41 ± 1.80 tokens — but for RL prototyping, we cap at 25 to accelerate training cycles and improve signal-to-noise in reward gradients.*

## 3️⃣ PPO, KL, and Entropy Stabilization
- **PPO loss** uses advantage clipping scaled with sequence length to prevent gradient spikes  
- **KL controller** adapts β more quickly and resets per curriculum update  
- **Entropy controller** adjusts targets based on sequence length to balance exploration  

---

## 🚀 Why ChemMiniQ3-SAbRLo?
- Prior approaches optimized only validity or physicochemical rules (Lipinski, etc.)  
- **Our method explicitly biases generation toward molecules that are not just valid, but also *easier to synthesize***  
- Extends beyond validity and rule-based rewards by explicitly biasing toward **synthetically accessible molecules**  
- The **symmetric curriculum + reward normalization** improves stability across varying sequence lengths  
- The **cyclical gradual curriculum + 25-token cap** potentially keeps training dynamic, avoids overfitting, and enables **<1hr RL policy iterations** on a single GPU  
- Shorter capped lengths (≤25 tokens) allow faster iteration, enabling more frequent updates and practical RL prototyping  
- Built from the ground up for (at least try to) **HuggingFace AutoModel/AutoTokenizer compatibility**

---

> 💡 **Target domain:** molecular generation (SELFIES).  
> 🔬 **Goal:** molecules that are valid, bioaware, and synthetically accessible.  
> 🚀 **Core innovation:** fast, modular prototyping of **MTP + RL fine-tuning pipelines** using standard HuggingFace components.

---

## Usage

- See `demo_usage.ipynb` or download it to use (I am still learning abt HF API so please be patient.)
  - model weights can be fetched [here](https://huggingface.co/gbyuvd/ChemMiniQ3-SAbRLo) if not automatically downloaded locally
- For training, clone this repo:
  - Customize config.json, run `train_withmtp.py` for NTP-to-MTP training
  - Run `train_ppokl_withsa.py` with either "chemq3" (bioaware-only no SA), "sa" (SA-only no bioaware), or "mix" (combined rewards)
- Dataset for training NTP/MTP can be fetched [here](https://huggingface.co/datasets/gbyuvd/sabrlo-chem-selfies-training)

## 🔮 Planned Experiments & Next Steps

We are actively working on scaling up ChemMiniQ3-SAbRLo with more ambitious experiments — **all designed for rapid iteration**:

- 📚 **Pretraining on a larger dataset** – up to **2.9M SELFIES molecules**  
- ⏱ **RL fine-tuning with extended steps** – test reward alignment speed under 25-token constraint  
- 🔬 **Comparative evaluation** – SA-only vs ChemQ3 vs Mix reward modes  
- 🧪 **Benchmarking** – validity, novelty, drug-likeness, and synthetic accessibility metrics  
- 🔄 **Automodel/AutoTokenizer integration** – verify full compatibility with HF ecosystem (e.g., `pipeline()`, `generate()`, `Trainer`)  
- 🧩 **Plug-and-play reward modules** – allow users to swap reward functions without touching model code  

---

## ❤️ Support the Project

Training and scaling require significant computational resources.  
If you’d like to support this research (e.g., helping us rent compute servers for rapid RL prototyping and MTP validation), you can contribute here:  

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/O4O710GFBZ) 

Every bit of support helps us push ChemMiniQ3-SAbRLo further! 🚀🧬

---

## To-Do
- **[ongoing]** Review, clean, and test train with existing codes
- [x] Warm up training on 163K dataset for MTP
- [ ] Complete pretraining on all ~1M dataset (when possible)
  - [x] Chunk I
  - [x] Chunk II
  - [x] Chunk III
  - [x] Chunk IV; early signs of overfitting with current lr 5e-5, will decrease to 3e-5
  - [ ] Chunk V
  - [ ] Chunk VI
- [ ] Publish complete pretraining on GitHub and HF (if compatible)
- **[ongoing]** Warm up PPO-RL with only Bioaware set on for 4500 steps
- [ ] Test and observe the stability of Mixed Rewards for 4500 steps
- [ ] Warm up PPO-RL with only SA set on for 7000 steps
- [ ] Complete RL fine-tuning on verified rewards system.
- [ ] Upload both warm-up MTP and PPO-RL models to HF repo
- **[ongoing]** Write demo blocks and demo JupyterNotebook on training from scratch and how to generate using pretrained model(s) 
- [ ] Ablation studies
- **[ongoing]** Implement and validate HF `AutoModel` and `AutoTokenizer` compatibility


---

## References
### BibTeX
#### Qwen3
```bibtex
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report}, 
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09388}, 
}
```

#### COCONUTDB
```bibtex
@article{sorokina2021coconut,
  title={COCONUT online: Collection of Open Natural Products database},
  author={Sorokina, Maria and Merseburger, Peter and Rajan, Kohulan and Yirik, Mehmet Aziz and Steinbeck, Christoph},
  journal={Journal of Cheminformatics},
  volume={13},
  number={1},
  pages={2},
  year={2021},
  doi={10.1186/s13321-020-00478-9}
}
```

#### ChemBL34
```bibtex
@article{zdrazil2023chembl,
  title={The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods},
  author={Zdrazil, Barbara and Felix, Eloy and Hunter, Fiona and Manners, Emma J and Blackshaw, James and Corbett, Sybilla and de Veij, Marleen and Ioannidis, Harris and Lopez, David Mendez and Mosquera, Juan F and Magarinos, Maria Paula and Bosc, Nicolas and Arcila, Ricardo and Kizil{\"o}ren, Tevfik and Gaulton, Anna and Bento, A Patr{\'i}cia and Adasme, Melissa F and Monecke, Peter and Landrum, Gregory A and Leach, Andrew R},
  journal={Nucleic Acids Research},
  year={2023},
  volume={gkad1004},
  doi={10.1093/nar/gkad1004}
}

@misc{chembl34,
  title={ChemBL34},
  year={2023},
  doi={10.6019/CHEMBL.database.34}
}
```

#### SuperNatural3
```bibtex
@article{Gallo2023,
  author = {Gallo, K and Kemmler, E and Goede, A and Becker, F and Dunkel, M and Preissner, R and Banerjee, P},
  title = {{SuperNatural 3.0-a database of natural products and natural product-based derivatives}},
  journal = {Nucleic Acids Research},
  year = {2023},
  month = jan,
  day = {6},
  volume = {51},
  number = {D1},
  pages = {D654-D659},
  doi = {10.1093/nar/gkac1008}
}
```

### Ranger21 Optimizer
``` bibtex
@article{wright2021ranger21,
      title={Ranger21: a synergistic deep learning optimizer}, 
      author={Wright, Less and Demeure, Nestor},
      year={2021},
      journal={arXiv preprint arXiv:2106.13731},
}
