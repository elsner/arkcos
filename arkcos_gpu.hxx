//
//    Copyright (C) 2010, 2011 Franz Elsner <f.elsner@mpa-garching.mpg.de>
//
//    This file is part of ARKCoS.
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program. If not, see <http://www.gnu.org/licenses/>.
//

#define PI_d 3.14159265358979323846
#define PI_f 3.1415926535f

void convolve_gpu(cufftReal*, cufftReal*, const double*, const double,
                  const double, const double, const float*,
                  const int, const int, const int, const int,
                  const int, const int);

void fft_map_gpu(cufftComplex*, cufftReal*, const int,
                 const int, const int, const int);

__global__ void rotate_fftrings_device(cufftComplex*, const float,
                                       const float, const int,
                                       const int, const int);

void convolve_with_kernel_gpu(cufftComplex*, cufftComplex*, cufftComplex*,
                              float*, cufftHandle, const double, const double,
                              const float, const int, const int,
                              const int, const int, const int);

__global__ void kernel2grid_gpu_device(float*, const double,
                                       const double, const float,
                                       const int, const int, const int,
                                       const int, const int);

__device__  __inline__ float interpolate_gpu_device(double,
                                                    const double,
                                                    const double);

__global__ void kernel_times_map_gpu_device(cufftComplex*, const cufftComplex*,
                                            const cufftComplex*, const int,
                                            const int, const int, const int,
                                            const int);

void ifft_map_gpu(cufftReal*, cufftComplex*, const int, const int,
                  const int, const int);
