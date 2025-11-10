#include "mex.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda_fp16.h> // For FP16 support
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define CUDA_CHECK(err) { cudaError_t err_ = (err); if (err_ != cudaSuccess) { mexPrintf("CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, cudaGetErrorString(err_)); mexErrMsgIdAndTxt("PCIimagingSparseCUDA:cudaError", cudaGetErrorString(err_)); } }
#define KERNEL_CHECK() { cudaError_t err_ = cudaGetLastError(); if (err_ != cudaSuccess) { mexPrintf("CUDA kernel error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, cudaGetErrorString(err_)); mexErrMsgIdAndTxt("PCIimagingSparseCUDA:kernelError", cudaGetErrorString(err_)); } }
#define CUFFT_CHECK(err) { cufftResult err_ = (err); if (err_ != CUFFT_SUCCESS) { mexPrintf("CUFFT error %d at %s:%d\n", err_, __FILE__, __LINE__); mexErrMsgIdAndTxt("PCIimagingSparseCUDA:cufftError", "CUFFT error"); } }

// --- Tuning parameters for RTX 4090 ---
#define NUM_STREAMS 8
#define BATCH_SIZE 1024   // Larger batch for better occupancy if memory allows
#define THREADS_PER_BLOCK 128 // Try 512 or 1024 if registers allow

// Maximum number of columns supported by the fixed-size arrays in kernels
#define MAX_KERNEL_COLS 128


// --- CUDA Kernels ---

__global__ void computeFftBinKernel(float* __restrict__ fftBin, int nfft, int binStart) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfft) {
        int shift = (nfft % 2 == 0) ? nfft / 2 : (nfft - 1) / 2;
        int srcIndex = (idx + shift) % nfft;
        float value = (float)srcIndex - (float)binStart;
        fftBin[idx] = (2.0f * M_PI * value) / (float)nfft;
    }
}

__global__ void realToComplexKernel(const float* __restrict__ realInput, cufftComplex* __restrict__ complexOutput, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        complexOutput[idx].x = realInput[idx];
        complexOutput[idx].y = 0.0f;
    }
}

__device__ cufftComplex cmul(cufftComplex a, cufftComplex b) {
    cufftComplex res;
    res.x = a.x * b.x - a.y * b.y;
    res.y = a.x * b.y + a.y * b.x;
    return res;
}

// Main kernel: one thread per pixel in batch (FP32 version)
__launch_bounds__(THREADS_PER_BLOCK, 2) // Min 2 blocks per SM for latency hiding
__global__ void PCI_beamform_batch_kernel(
    const cufftComplex* __restrict__ d_RFFT, // [nfft x ncols]
    const float* __restrict__ d_fftBin,           // [nfft]
    const float* __restrict__ d_all_delays,       // [ncols x total_pixels]
    int nfft, int ncols, int total_pixels,
    int batch_start_idx, int batch_size,
    int FlowerIndex, int freq_band_len,
    float p_param, float inv_p_param, int check, // Added inv_p_param
    float* __restrict__ d_output                  // [total_pixels]
) {
    int batch_pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_pixel >= batch_size) return;
    int pixel_idx = batch_start_idx + batch_pixel;
    if (pixel_idx >= total_pixels) return;

    const float* delays = d_all_delays + pixel_idx * ncols;
    float pixel_sum = 0.0f;

    float var_x[MAX_KERNEL_COLS], var_y[MAX_KERNEL_COLS];

    for (int f = 0; f < freq_band_len; ++f) {
        int r = FlowerIndex + f;

        // --- Build var for this frequency: phase-shifted, all channels ---
        for (int c = 0; c < ncols; ++c) {
            float angle = -delays[c] * d_fftBin[r];
            float cs, sn;
            __sincosf(angle, &sn, &cs); // compute sin and cos together
            int rfft_idx = r + c * nfft; 
            cufftComplex val = d_RFFT[rfft_idx];
            var_x[c] = val.x * cs - val.y * sn;
            var_y[c] = val.x * sn + val.y * cs;
        }

        if (check == 1) {
            // --- pCF branch ---
            float sum_real_das_comp = 0.0f, sum_imag_das_comp = 0.0f;
            for (int c = 0; c < ncols; ++c) {
                //  Direct sum of phase-shifted components for pDAS part of pCF
                sum_real_das_comp += var_x[c];
                sum_imag_das_comp += var_y[c];
            }
            float pDAS_val = hypotf(sum_real_das_comp, sum_imag_das_comp);

            float sum_real_p = 0.0f, sum_imag_p = 0.0f, Dr = 0.0f;
            for (int c = 0; c < ncols; ++c) {
                float mag = hypotf(var_x[c], var_y[c]);
                float phase = atan2f(var_y[c], var_x[c]); 
                float comp = powf(mag, inv_p_param);    // use precomputed inv_p_param
                sum_real_p += comp * __cosf(phase);     
                sum_imag_p += comp * __sinf(phase);
                Dr += mag * mag; 
            }
            float abs_sum_p = hypotf(sum_real_p, sum_imag_p);
            float beamformed = powf(abs_sum_p, p_param);
            float Nr = beamformed * beamformed;
            if (Dr == 0.0f) Dr = 1e-12f; 
            float CF = (1.0f / ncols) * (Nr / Dr);

            float DCoffset = 0.0f;
            for (int c = 0; c < ncols; ++c) {
                float real_cf = var_x[c] * CF;
                float imag_cf = var_y[c] * CF;
                DCoffset += real_cf * real_cf + imag_cf * imag_cf;
            }

            float val = pDAS_val * CF;
            pixel_sum += (val * val) - DCoffset;
        } else {
            // --- pDAS branch ---
            float sum_real = 0.0f, sum_imag = 0.0f;
            for (int c = 0; c < ncols; ++c) {
                float mag = hypotf(var_x[c], var_y[c]);
                float phase = atan2f(var_y[c], var_x[c]);
                float comp = powf(mag, inv_p_param); // use precomputed inv_p_param
                sum_real += comp * __cosf(phase);
                sum_imag += comp * __sinf(phase);
            }
            float pDAS = powf(hypotf(sum_real, sum_imag), p_param);

            float DCoffset = 0.0f;
            for (int c = 0; c < ncols; ++c) {
                DCoffset += var_x[c] * var_x[c] + var_y[c] * var_y[c];
            }
            pixel_sum += (pDAS * pDAS) - DCoffset;
        }
    }
    d_output[pixel_idx] = pixel_sum;
}

// Main kernel: one thread per pixel in batch (FP16 version)
__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void PCI_beamform_batch_kernel_fp16(
    const cufftComplex* __restrict__ d_RFFT, 
    const float* __restrict__ d_fftBin,          
    const float* __restrict__ d_all_delays,      
    int nfft, int ncols, int total_pixels,
    int batch_start_idx, int batch_size,
    int FlowerIndex, int freq_band_len,
    float p_param, float inv_p_param, int check, // Takes inv_p_param for consistency
    float* __restrict__ d_output                 
) {
    int batch_pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_pixel >= batch_size) return;
    int pixel_idx = batch_start_idx + batch_pixel;
    if (pixel_idx >= total_pixels) return;

    const float* delays = d_all_delays + pixel_idx * ncols;
    __half pixel_sum_h = __float2half(0.0f);

    __half var_x[MAX_KERNEL_COLS], var_y[MAX_KERNEL_COLS];

    for (int f = 0; f < freq_band_len; ++f) {
        int r = FlowerIndex + f;
        for (int c = 0; c < ncols; ++c) {
            float angle = -delays[c] * d_fftBin[r];
            float cs, sn;
            __sincosf(angle, &sn, &cs); 
            int rfft_idx = r + c * nfft;
            cufftComplex val = d_RFFT[rfft_idx]; 
            var_x[c] = __float2half(val.x * cs - val.y * sn);
            var_y[c] = __float2half(val.x * sn + val.y * cs);
        }

        if (check == 1) {
            __half sum_real_das_comp_h = __float2half(0.0f);
            __half sum_imag_das_comp_h = __float2half(0.0f);
            for (int c = 0; c < ncols; ++c) {
                sum_real_das_comp_h = __hadd(sum_real_das_comp_h, var_x[c]);
                sum_imag_das_comp_h = __hadd(sum_imag_das_comp_h, var_y[c]);
            }
            float pDAS_val = hypotf(__half2float(sum_real_das_comp_h), __half2float(sum_imag_das_comp_h));

            __half sum_real_p_h = __float2half(0.0f);
            __half sum_imag_p_h = __float2half(0.0f);
            __half Dr_h = __float2half(0.0f);
            for (int c = 0; c < ncols; ++c) {
                float var_x_f = __half2float(var_x[c]);
                float var_y_f = __half2float(var_y[c]);
                float mag_f = hypotf(var_x_f, var_y_f);
                float phase_f = atan2f(var_y_f, var_x_f);
                float comp_f = powf(mag_f, inv_p_param); 
                
                sum_real_p_h = __hadd(sum_real_p_h, __float2half(comp_f * cosf(phase_f))); 
                sum_imag_p_h = __hadd(sum_imag_p_h, __float2half(comp_f * sinf(phase_f))); 
                Dr_h = __hadd(Dr_h, __float2half(mag_f * mag_f));
            }
            float abs_sum_p_f = hypotf(__half2float(sum_real_p_h), __half2float(sum_imag_p_h));
            float beamformed_f = powf(abs_sum_p_f, p_param);
            float Nr_f = beamformed_f * beamformed_f;
            float Dr_f_val = __half2float(Dr_h);
            if (Dr_f_val == 0.0f) Dr_f_val = 1e-12f;
            float CF_f = (1.0f / ncols) * (Nr_f / Dr_f_val);

            __half DCoffset_h = __float2half(0.0f);
            for (int c = 0; c < ncols; ++c) {
                float var_x_f = __half2float(var_x[c]);
                float var_y_f = __half2float(var_y[c]);
                float real_cf_f = var_x_f * CF_f;
                float imag_cf_f = var_y_f * CF_f;
                DCoffset_h = __hadd(DCoffset_h, __float2half(real_cf_f * real_cf_f + imag_cf_f * imag_cf_f));
            }
            float val_f = pDAS_val * CF_f;
            pixel_sum_h = __hadd(pixel_sum_h, __float2half((val_f * val_f) - __half2float(DCoffset_h)));
        } else {
            __half sum_real_h = __float2half(0.0f);
            __half sum_imag_h = __float2half(0.0f);
            for (int c = 0; c < ncols; ++c) {
                float var_x_f = __half2float(var_x[c]);
                float var_y_f = __half2float(var_y[c]);
                float mag_f = hypotf(var_x_f, var_y_f);
                float phase_f = atan2f(var_y_f, var_x_f);
                float comp_f = powf(mag_f, inv_p_param); 

                sum_real_h = __hadd(sum_real_h, __float2half(comp_f * cosf(phase_f)));
                sum_imag_h = __hadd(sum_imag_h, __float2half(comp_f * sinf(phase_f)));
            }
            float pDAS_f = powf(hypotf(__half2float(sum_real_h), __half2float(sum_imag_h)), p_param);

            __half DCoffset_h = __float2half(0.0f);
            for (int c = 0; c < ncols; ++c) {
                float var_x_f = __half2float(var_x[c]);
                float var_y_f = __half2float(var_y[c]);
                DCoffset_h = __hadd(DCoffset_h, __float2half(var_x_f * var_x_f + var_y_f * var_y_f));
            }
            pixel_sum_h = __hadd(pixel_sum_h, __float2half((pDAS_f * pDAS_f) - __half2float(DCoffset_h)));
        }
    }
    d_output[pixel_idx] = __half2float(pixel_sum_h);
}

// --- MEX Gateway Function ---
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 11) mexErrMsgIdAndTxt("PCIimagingSparseCUDA:nrhs", 
        "Eleven inputs required: RF_Arr, element_Pos_Array_um, speed_Of_Sound_umps, "
        "RF_Start_Time, sampling_Freq, image_Range_X_um, image_Range_Z_um, p, "
        "all_delays_matrix, range_frq, check");

    // --- Get Inputs & Dimensions ---
    const float* h_RF_Arr = (const float*)mxGetData(prhs[0]);
    size_t nfft = mxGetM(prhs[0]);
    size_t ncols = mxGetN(prhs[0]);

    if (ncols > MAX_KERNEL_COLS) {
        mexErrMsgIdAndTxt("PCIimagingSparseCUDA:ncolsError", 
                          "Number of columns (ncols=%zu) exceeds MAX_KERNEL_COLS (%d). "
                          "Kernels use fixed-size arrays. Recompile with larger MAX_KERNEL_COLS.",
                          ncols, MAX_KERNEL_COLS);
    }

    size_t total_pixels = mxGetN(prhs[8]); 
    const float* h_all_delays = (const float*)mxGetData(prhs[8]);
    float p_param = (float)mxGetScalar(prhs[7]);
    float inv_p_param = 1.0f / p_param; // Will produce INF if p_param is 0.0f, which powf handles.

    float* range_frq = (float*)mxGetData(prhs[9]);
    int check = (int)mxGetScalar(prhs[10]);
    float sampling_Freq = (float)mxGetScalar(prhs[4]);
    
    mwSize numX_mw = (mwSize)mxGetNumberOfElements(prhs[5]);
    mwSize numZ_mw = (mwSize)mxGetNumberOfElements(prhs[6]);

    // --- Frequency band indices ---
    size_t FlowerIndex = 0;
    size_t FupperIndex = 0; 
    size_t freq_band_len = 0;

    if (nfft > 0) {
        std::vector<float> fk(nfft);
        for (size_t i = 0; i < nfft; ++i) fk[i] = ((float)i) * sampling_Freq / (float)nfft;

        // Determine FlowerIndex (closest to range_frq[0])
        FlowerIndex = 0; // Default to first element
        float minDiff_lower = fabsf(fk[0] - range_frq[0]);
        for (size_t i = 1; i < nfft; ++i) {
            float current_diff = fabsf(fk[i] - range_frq[0]);
            if (current_diff < minDiff_lower) {
                FlowerIndex = i;
                minDiff_lower = current_diff;
            }
        }
        
        // Determine FupperIndex (closest to range_frq[1])
        FupperIndex = 0; // Default to first element
        float minDiff_upper = fabsf(fk[0] - range_frq[1]);
        for (size_t i = 1; i < nfft; ++i) {
            float current_diff = fabsf(fk[i] - range_frq[1]);
            if (current_diff < minDiff_upper) {
                FupperIndex = i;
                minDiff_upper = current_diff;
            }
        }

        if (FupperIndex >= FlowerIndex) {
            freq_band_len = FupperIndex - FlowerIndex + 1;
        } else {
            freq_band_len = 0; // Avoid issues with size_t underflow if band is inverted
            mexWarnMsgIdAndTxt("PCIimagingSparseCUDA:freqOrder", 
                               "FupperIndex (%zu) is less than FlowerIndex (%zu). "
                               "This implies an empty or invalid frequency band. Setting band length to 0. "
                               "Check input 'range_frq'.", FupperIndex, FlowerIndex);
        }
    } else { // nfft is 0
        FlowerIndex = 0;
        FupperIndex = 0;
        freq_band_len = 0;
    }
    
    mexPrintf("Starting CUDA PCI Imaging (p=%.2f, check=%d, freq band [%zu:%zu], length=%zu)\n", 
              p_param, check, FlowerIndex, FupperIndex, freq_band_len);

    plhs[0] = mxCreateNumericMatrix(numX_mw, numZ_mw, mxSINGLE_CLASS, mxREAL);
    float* h_beamformed_Image = (float*)mxGetData(plhs[0]);
    if (nlhs > 1) { 
        plhs[1] = mxCreateDoubleScalar(0.0); 
    }

    // --- Allocate Device Memory ---
    cufftComplex *d_RFFT = nullptr;
    float *d_fftBin = nullptr;
    float *d_beamformed_Image = nullptr;
    float *d_all_delays = nullptr;

    CUDA_CHECK(cudaMalloc(&d_beamformed_Image, total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_beamformed_Image, 0, total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_all_delays, ncols * total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_all_delays, h_all_delays, ncols * total_pixels * sizeof(float), cudaMemcpyHostToDevice));
    
    if (nfft > 0) { // Only allocate FFT related memory if nfft > 0
        CUDA_CHECK(cudaMalloc(&d_RFFT, nfft * ncols * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_fftBin, nfft * sizeof(float)));
    }


    // --- Precomputations ---
    if (nfft > 0) {
        int blockSize_fftbin = THREADS_PER_BLOCK; 
        int gridSize_fftbin = (nfft + blockSize_fftbin - 1) / blockSize_fftbin;
        int binStart = nfft / 2; 
        computeFftBinKernel<<<gridSize_fftbin, blockSize_fftbin>>>(d_fftBin, nfft, binStart);
        KERNEL_CHECK();
    }

    if (nfft > 0 && ncols > 0) {
        cufftComplex* d_RData_complex_temp = nullptr;
        float* d_RF_Arr_temp = nullptr; 
        size_t total_rf_elements = nfft * ncols;

        CUDA_CHECK(cudaMalloc(&d_RData_complex_temp, total_rf_elements * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_RF_Arr_temp, total_rf_elements * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_RF_Arr_temp, h_RF_Arr, total_rf_elements * sizeof(float), cudaMemcpyHostToDevice));
        
        int blockSize_real2c = THREADS_PER_BLOCK;
        int gridSize_real2c = (total_rf_elements + blockSize_real2c - 1) / blockSize_real2c;
        realToComplexKernel<<<gridSize_real2c, blockSize_real2c>>>(d_RF_Arr_temp, d_RData_complex_temp, total_rf_elements);
        KERNEL_CHECK();
        CUDA_CHECK(cudaFree(d_RF_Arr_temp)); 

        cufftHandle plan_forward;
        CUFFT_CHECK(cufftPlan1d(&plan_forward, nfft, CUFFT_C2C, ncols));
        CUFFT_CHECK(cufftExecC2C(plan_forward, d_RData_complex_temp, d_RFFT, CUFFT_FORWARD));
        CUFFT_CHECK(cufftDestroy(plan_forward));
        CUDA_CHECK(cudaFree(d_RData_complex_temp)); 
    } else if (freq_band_len > 0) { // nfft or ncols is zero but we need to process frequencies
         mexWarnMsgIdAndTxt("PCIimagingSparseCUDA:ZeroSizeFFT", 
                            "nfft or ncols is zero, but freq_band_len > 0. "
                            "FFT step skipped. d_RFFT will contain uninitialized data, likely leading to errors.");
    }

    // --- Create multiple CUDA streams ---
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // --- Launch Batching Kernel in different streams ---
    if (total_pixels > 0 && freq_band_len > 0 && nfft > 0 && ncols > 0) { 
        int batch_size_local = BATCH_SIZE; 
        int threadsPerBlock_local = THREADS_PER_BLOCK; 

        for (size_t batch_start_idx = 0; batch_start_idx < total_pixels; batch_start_idx += batch_size_local) {
            int stream_idx = (batch_start_idx / batch_size_local) % NUM_STREAMS;
            int current_batch_size = std::min(batch_size_local, (int)(total_pixels - batch_start_idx));
            int blocksPerGrid = (current_batch_size + threadsPerBlock_local - 1) / threadsPerBlock_local;

            PCI_beamform_batch_kernel<<<blocksPerGrid, threadsPerBlock_local, 0, streams[stream_idx]>>>(
                d_RFFT, d_fftBin, d_all_delays, nfft, ncols, total_pixels,
                batch_start_idx, current_batch_size,
                FlowerIndex, freq_band_len, p_param, inv_p_param, check, d_beamformed_Image
            );
            KERNEL_CHECK(); 
        }
    } else if (total_pixels > 0 && freq_band_len > 0) {
        // This case means nfft or ncols was 0, but we still tried to image. Output will be 0.
         mexWarnMsgIdAndTxt("PCIimagingSparseCUDA:NoImageData", 
                            "total_pixels > 0 and freq_band_len > 0, but nfft or ncols is 0. "
                            "Beamforming kernel not launched. Output image will be all zeros.");
    }


    // --- Synchronize and clean up streams ---
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    // --- Copy Result Back to Host ---
    if (total_pixels > 0) {
      CUDA_CHECK(cudaMemcpy(h_beamformed_Image, d_beamformed_Image, total_pixels * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // --- Cleanup ---
    if (d_RFFT) CUDA_CHECK(cudaFree(d_RFFT));
    if (d_fftBin) CUDA_CHECK(cudaFree(d_fftBin));
    if (d_beamformed_Image) CUDA_CHECK(cudaFree(d_beamformed_Image));
    if (d_all_delays) CUDA_CHECK(cudaFree(d_all_delays));

    mexPrintf("CUDA PCI Imaging complete.\n");
}