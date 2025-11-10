# pth-root-coherence-factor  
GPU-accelerated implementation of the **p-th root coherence factor (pCF)** beamforming method for ultrasound imaging, developed at **MUSE Lab, IIT Gandhinagar**.

---

## Overview  
This repository provides **CUDA and MATLAB implementations** of the *p-th root coherence factor* (pCF) beamforming algorithm.  
The method combines **p-norm Delay-And-Sum (pDAS)** with a **coherence factor (CF)** to improve contrast and resolution in **plane-wave ultrasound imaging**.

---

##  Motivation  
Traditional Delay-And-Sum (DAS) beamforming coherently sums channel signals but can suffer from low contrast in noisy conditions.  
Coherence Factor (CF) weighting helps emphasize coherent echoes while suppressing incoherent noise.  


Combining p-norm DAS with CF weighting (pCF) provides tunable control between **resolution**, **contrast**, and **speckle suppression**.

---

## 📁 Repository Structure  

| File | Description |
|------|--------------|
| `PCI_cuda_single.cu` / `PCI_cuda_double.cu` | CUDA source files for single- and double-precision kernels |
| `PCIfreqRBC_CPU.m`, `PCIfreqRBC_GPU_single.m`, `PCIfreqRBC_GPU_double.m` | MATLAB driver scripts for CPU & GPU versions |
| `pthcoherencefactorfreq.m`, `pthrootfreq.m`, `simpledelayfreq.m` | MATLAB implementations of pCF, pDAS, and DAS beamforming |
| `Image_generation_code.m` | Example MATLAB image reconstruction script |
| `example_data.mat`, `example_image.png` | Example datasets and output images |
| `README.md` | This documentation file |

---

## ⚙️ Requirements  
- **NVIDIA GPU** with CUDA support (Compute Capability ≥ 5.0)  
- **CUDA Toolkit** installed (tested on CUDA 11.x / 12.x)  
- **MATLAB** R2021a or later  
- **Parallel Computing Toolbox** (recommended)  

For double-precision kernels, GPUs with strong FP64 performance (e.g., RTX A6000, A100, etc.) are preferred.

---

## 🧩 Installation & Build  

### 1. Clone the repository  
```bash
git clone https://github.com/muselab-iitgn/pth-root-coherence-factor.git
cd pth-root-coherence-factor
