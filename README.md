# pth Root Coherence Factor  
GPU-accelerated implementation of the **p-th root coherence factor (pCF)** beamforming method for ultrasound imaging, developed at **MUSE Lab, IIT Gandhinagar**.

---

## Overview  
This repository provides **CUDA and MATLAB implementations** of the *p-th root coherence factor* (pCF) beamforming algorithm.  
The method combines **p-norm Delay-And-Sum (pDAS)** with a **coherence factor (CF)** to improve contrast and resolution in **plane-wave ultrasound imaging**.


---

##  Repository Structure  

| File | Description |
|------|--------------|
| `PCI_cuda_single.cu` / `PCI_cuda_double.cu` | CUDA source files for single- and double-precision kernels |
| `PCIfreqRBC_CPU.m`, `PCIfreqRBC_GPU_single.m`, `PCIfreqRBC_GPU_double.m` | MATLAB driver scripts for CPU & GPU versions |
| `pthcoherencefactorfreq.m`, `pthrootfreq.m`, `simpledelayfreq.m` | MATLAB implementations of pCF, pDAS, and DAS beamforming |
| `Image_generation_code.m` | Example MATLAB image reconstruction script |
| `PData_RBC2_2023Feb14_1_area_2_dataset_99_66338.mat`, `PCI_image_dB_scale.png` | Example datasets and output images |
| `README.md` | This documentation file |

---

##  Requirements  
- **NVIDIA GPU** with CUDA support (Compute Capability ≥ 5.0)  
- **CUDA Toolkit** installed (tested on CUDA 11.x / 12.x)  
- **MATLAB** R2021a or later  
- **Parallel Computing Toolbox** (recommended)  

---

##  Installation & Build  

### 1. Clone the repository  
```bash
git clone https://github.com/muselab-iitgn/pth-root-coherence-factor.git
cd pth-root-coherence-factor
```

### 2. Compile CUDA kernels  
 If you want to change configuration for GPU (e.g., threads per block), modify the relevant parameters in the CUDA source files before recompiling. Otherwise, precompiled MEX files are provided for reference.
```bash
# For single precision
mexcuda -lcufft PCI_cuda_single.cu 

# For double precision  
mexcuda -lcufft PCI_cuda_double.cu 
```



### 3. Verify MATLAB GPU support  
```matlab
% Check GPU availability
gpuDevice
```

---

## Usage  
1. Add the path to dataset in `Image_generation_code.m`.  
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
4. Run the `Image_generation_code.m` script to generate and visualise images.
---




## License  

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.


---
