# **E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models**

This repository provides the official implementation of **E-BATS**, the first **backpropagation-free test-time adaptation (TTA)** framework designed explicitly for **Speech Foundation Models (SFMs)** such as **Wav2Vec2** and **HuBERT**.  

For questions or issues, please email: **jiahengdong215@gmail.com**

---

## 📘 **Table of Contents**
1. [Introduction](#introduction)  
2. [Key Features](#key-features)  
3. [Installation](#installation)  
5. [How to Run](#how-to-run)  
   - [Benchmark Framework (TTAs-1)](#benchmark-framework-ttas-1)  
   - [Standalone E-BATS (TTAs-2)](#standalone-e-bats-ttas-2)  
6. [Datasets](#datasets)   
8. [Citation](#citation)

---

## 🧠 **Introduction**

**E-BATS** introduces an **efficient single-utterance test-time adaptation framework** for speech recognition under real-world acoustic domain shifts.  
Traditional test-time adaptation methods rely on **backpropagation** for speech tasks, which is **memory-intensive** and unsuitable for speech foundation models or on-device inference.  
E-BATS solves this by combining:

1. **Lightweight Prompt Adaptation (LPA)** — a derivative-free, forward-only tuning mechanism.  
2. **Multi-Scale Loss Function** — aligns global (utterance-level) and local (token-level) features.  
3. **Test-Time Exponential Moving Average (T-EMA)** — stabilizes adaptation across utterances.

These innovations enable **robust and memory-efficient adaptation** of speech models during inference without source data or labels.

---

## 🚀 **Key Features**

- 🔹 **Backpropagation-Free Adaptation**: Uses CMA-ES for optimization without gradient computation.  
- 🔹 **Prompt-Based Adaptation**: Introduces lightweight learnable prompts to adjust CNN latent features.  
- 🔹 **Multi-Scale Loss**: Combines entropy minimization, utterance-level, and token-level alignment.  
- 🔹 **T-EMA Module**: Smooths adaptation across streaming utterances to prevent instability.  
- 🔹 **Unified Benchmark Framework**: Includes multiple state-of-the-art (SOTA) TTA methods for fair evaluation.  
- 🔹 **Extensive Dataset Support**: Tested across **LibriSpeech**, **CHiME-3**, **TED-LIUM**, and **CommonVoice** under sixteen acoustic conditions.  
- 🔹 **Highly Memory-Efficient**: Up to **6.4× GPU memory reduction** vs. BP-based methods.

---

## ⚙️ **Installation**

### **Prerequisites**
- Python ≥ 3.8  
- CUDA-enabled GPU (recommended)  
- PyTorch ≥ 1.12  
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

### **Setup**
```bash
git clone https://github.com/<your-repo>/E-BATS.git
cd E-BATS
pip install -r requirements.txt
```

---

## 🏃 **How to Run**

This repository provides two frameworks for running test-time adaptation experiments:

### **Benchmark Framework (TTAs-1)**

The **TTAs-1** directory contains a unified benchmark framework supporting multiple TTA methods including SUTA, TENT, EATA, CEA, T3A, and more.

#### **Basic Usage**

```bash
cd TTAs-1
python run_benchmark.py \
    --tta_name suta \
    --dataset_name librispeech \
    --config config/system/bp-based.yaml \
    --tta_config config/tta/suta.yaml \
    --split test-other \
    --path /path/to/LibriSpeech \
    --batch_size 1 \
    --extra_noise 0.02 \
```

#### **Configuration Files**

- **System Config** (`config/system/`): Defines model architecture and optimization settings
- **TTA Config** (`config/tta/`): Specifies method-specific hyperparameters

---

### **Standalone E-BATS (TTAs-2)**

The **TTAs-2** directory contains standalone implementations of E-BATS and other advanced TTA methods with full control over all parameters.

#### **Basic Usage**

```bash
cd TTAs-2
python E-BATS.py 
    --asr facebook/hubert-large-ls960-ft \
    --steps 25 \
    --pop_size 50 \
    --batch_size 1 \
    --dataset_name chime \
    --early_stop_threshold 0.001\
    --patient 3\
    --dataset_dir /home/jiahengd/tta-suta/CHiME3 \
    --extra_noise 0.0 \
    --beta 2.0\
    --alpha 1.0\
    --temp 2.0\
    --split str-simu \
    --reset_frequency 1 \
    --ema_decay 0.8 \
    --use_tema \
    --covariance_ratio 1.0 \
    --random_seed 2024 \
    --confidence_max 5.0 \
    --tokenwise \
```

---

## 📊 **Datasets**

### **LibriSpeech**
Clean and noisy speech data from audiobooks.
- **Splits**: `test-clean`, `test-other`
- **Download**: [OpenSLR](http://www.openslr.org/12/)

### **CHiME-3**
Real and simulated noisy speech in various acoustic environments.
- **Environments**: bus, cafe, pedestrian area, street (each with real and simulated versions)
- **Splits**: `et05_bus_real`, `et05_bus_simu`, `et05_caf_real`, `et05_caf_simu`, `et05_ped_real`, `et05_ped_simu`, `et05_str_real`, `et05_str_simu`
- **Download**: [CHiME-3](https://www.chimechallenge.org/challenges/chime3/data.html#Download)

### **CommonVoice**
Multi-speaker, multi-recording-background speech.
- **Download**: [Mozilla CommonVoice](https://commonvoice.mozilla.org/)

### **TED-LIUM**
TED talk recordings.
- **Download**: [OpenSLR](http://www.openslr.org/51/)

---

## 📄 **Citation**

If you use this code in your research, please cite:

```bibtex
@inproceedings{
  anonymous2025ebats,
  title={E-{BATS}: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models},
  author={Jiaheng Dong, Hong Jia, Soumyajit Chatterjee, Abhirup Ghosh, James Bailey, Ting Dang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=WwzurufeFN}
}
```