function [beamformed_Image,delay] = PCIfreqRBC_GPU_single(RF_Arr, element_Pos_Array_um_X, speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, image_Range_X_um, image_Range_Z_um,p,range_frq,check)
% Performs PCI beamforming using a CUDA MEX function with precomputed delays.

    % Get dimensions
    [nfft, ncols] = size(RF_Arr); % nfft=Axial_depth, ncols=number of elements
    numX = length(image_Range_X_um); % Number of rows in beamformed image
    numZ = length(image_Range_Z_um); % Number of columns in beamformed image
    total_pixels = numX * numZ;

    % --- Precompute all delays on the host ---
    tic;

    all_delays_matrix = zeros(ncols, total_pixels); % Preallocate (ncols x total_pixels)
    pixel_index = 0; % Linear pixel index (1-based, column-major)

    % Loop Z first, then X to match MATLAB's column-major linearization
    for zi = 1:numZ
        currentZ_um = image_Range_Z_um(zi);
        for xi = 1:numX
            pixel_index = pixel_index + 1;
            currentX_um = image_Range_X_um(xi);

            % Calculate distance from pixel (xi, zi) to each element's position
            dist_um = sqrt( (currentX_um - element_Pos_Array_um_X(1,:)).^2 + ...
                            (currentZ_um - element_Pos_Array_um_X(2,:)).^2 ); 

            % Calculate time-of-flight
            time_s = dist_um / speed_Of_Sound_umps; 

            % Calculate delay in samples (negative for focusing/DAS)
            delay_samples = -time_s * sampling_Freq; 

            % Store the delay vector for this pixel as a column
            all_delays_matrix(:, pixel_index) = delay_samples'; 
        end
    end
    precomputation_time = toc;

    tic;
    beamformed_Image = PCI_cuda_single( ...
    single(RF_Arr), ...                    % [nfft x ncols] real double
    single(element_Pos_Array_um_X), ...    % [2 x ncols] real double (not used, but required for compatibility)
    single(speed_Of_Sound_umps), ...       % scalar double (not used, but required for compatibility)
    single(RF_Start_Time), ...             % scalar double (not used, but required for compatibility)
    single(sampling_Freq), ...             % scalar double
    single(image_Range_X_um), ...          % [numX x 1] real double
    single(image_Range_Z_um), ...          % [numZ x 1] real double
    single(p), ...                         % scalar double (p-th root)
    single(all_delays_matrix), ...         % [ncols x (numX*numZ)] real double, precomputed delays (see below)
    single(range_frq), ...                 % [1x2] double, frequency range [f_low, f_high]
    single(check) ...                      % scalar double, 1 for pCF, 0 for pDAS
);

    
    delay = toc;
    delay=delay+precomputation_time;
    beamformed_Image=double(beamformed_Image);

end 