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

##  Repository Structure  

| File | Description |
|------|--------------|
| `PCI_cuda_single.cu` / `PCI_cuda_double.cu` | CUDA source files for single- and double-precision kernels |
| `PCIfreqRBC_CPU.m`, `PCIfreqRBC_GPU_single.m`, `PCIfreqRBC_GPU_double.m` | MATLAB driver scripts for CPU & GPU versions |
| `pthcoherencefactorfreq.m`, `pthrootfreq.m`, `simpledelayfreq.m` | MATLAB implementations of pCF, pDAS, and DAS beamforming |
| `Image_generation_code.m` | Example MATLAB image reconstruction script |
| `example_data.mat`, `example_image.png` | Example datasets and output images |
| `README.md` | This documentation file |

---

##  Requirements  
- **NVIDIA GPU** with CUDA support (Compute Capability ≥ 5.0)  
- **CUDA Toolkit** installed (tested on CUDA 11.x / 12.x)  
- **MATLAB** R2021a or later  
- **Parallel Computing Toolbox** (recommended)  

For double-precision kernels, GPUs with strong FP64 performance (e.g., RTX A6000, A100, etc.) are preferred.

---

##  Installation & Build  

### 1. Clone the repository  
```bash
git clone https://github.com/muselab-iitgn/pth-root-coherence-factor.git
cd pth-root-coherence-factor
```

### 2. Compile CUDA kernels  
```bash
# For single precision
mexcuda -lcufft PCI_cuda_single.cu 

# For double precision  
mexcuda -lcufft PCI_cuda_double.cu 
```
 If you want to change configuration for GPU (e.g., threads per block), modify the relevant parameters in the CUDA source files before recompiling. Otherwise, precompiled MEX files are provided for reference.


### 3. Verify MATLAB GPU support  
```matlab
% Check GPU availability
gpuDevice
```

---

## Usage  
1. Add the path to dataset in Image_generation_code.m.  
2. Set imaging parameters (element positions, speed of sound, sampling frequency, etc.).  
3. Call the desired beamforming function:  
```matlab
% For CPU version
image_CPU = PCIfreqRBC_CPU(RF_data, element_Pos_Array_X, ...
    speed_Of_Sound, RF_Start_Time, sampling_Freq, ...
    image_Range_X, image_Range_Z, p_value, range_frq, cf_weight);    
% For GPU single precision  
image_GPU_single = PCIfreqRBC_GPU_single(RF_data, element_Pos_Array_X, ...
    speed_Of_Sound, RF_Start_Time, sampling_Freq, ...
    image_Range_X, image_Range_Z, p_value, range_frq, cf_weight);    
% For GPU double precision  
image_GPU_double = PCIfreqRBC_GPU_double(RF_data, element_Pos_Array_X, ...
    speed_Of_Sound, RF_Start_Time, sampling_Freq, ...
    image_Range_X, image_Range_Z, p_value, range_frq, cf_weight);    
```
---

## Algorithm Details  

### p-th Root Coherence Factor Formula  
The pCF beamformed output combines p-norm DAS with coherence factor weighting:

```
y[n] = (CF[n])^α × (∑|x_i[n + τ_i]|^p)^(1/p)

where:
- CF[n]: Coherence factor at sample n
- α: CF weighting parameter (cf_weight)  
- p: p-norm parameter (p_value)
- x_i[n]: i-th channel signal
- τ_i: Time delay for i-th channel
```

### Coherence Factor Calculation  
```
CF[n] = (∑x_i[n + τ_i])² / (N × ∑x_i[n + τ_i]²)

where N is the number of active channels
```

---

## Validation & Testing  

### Run Built-in Tests  
```matlab
% Test different beamforming methods
Image_generation_code;

% Compare CPU vs GPU results
test_cpu_gpu_consistency;
```

---


---

## License  

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

---
