# NanyinHGNN
# 🎵 NanyinHGNN

**Oral Tradition-Encoded NanyinHGNN: Integrating Nanyin Music Preservation and Generation through a Pipa-Centric Dataset**  
📄 [arXiv:2510.26817](https://arxiv.org/abs/2510.26817)  
📘 CSMT 2025, 12th National Conference on Sound and Music Technology  

my personal web:  (https://wishzhai.github.io)

---

## 🧠 Overview

**NanyinHGNN** is a hierarchical graph neural network designed to integrate *music preservation* and *generation* tasks for **Nanyin**, a UNESCO-recognized traditional Chinese music form.  
By encoding oral-tradition-informed relationships within a **Pipa-centric dataset**, the model bridges symbolic heterophonic ornamentation and deep generative modeling.

This repository provides:
- Dataset preparation and graph construction tools  
- Two-stage training and generation scripts  
- Example checkpoints and configuration files  

---

## ⚙️ Environment Setup

### 1. Create a new conda environment
```bash
conda create -n nanyin_hgnn python=3.10
conda activate nanyin_hgnn

# CUDA 11.8
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia 
pytorch_lightning==2.2
 # If you have installed dgl-cuXX package, please uninstall it first. 
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html                                                  
pip install numpy==1.22.4

python scripts/generate_enhanced.py --model 1.ckpt --output generated_music --tempo 100 --temperature 0.8
python train_two_stage.py --config configs/two_stage_training.yaml --batch_size 2 --num_workers 2 --current_stage 1
python train_two_stage.py --config configs/two_stage_training.yaml --batch_size 2 --num_workers 2 --current_stage 2 --resume_from_checkpoint nanyin_model_stage1_20250328_033403.ckpt
