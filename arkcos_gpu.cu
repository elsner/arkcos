//
//    Copyright (C) 2010, 2011, 2018 Franz Elsner <f.elsner@mpa-garching.mpg.de>
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

#include <cufft.h>
#include <helper_cuda.h>
#include "arkcos_gpu.hxx"

__constant__ float                                          kernel_gpu__alpha[8000];
__constant__ double                                         theta_gpu__ring[4100];

//-----------------------------------------------------------------------
// Wrapper function for convolution
//-----------------------------------------------------------------------

void convolve_gpu(cufftReal* map_out__pix,
                  cufftReal* map_in__pix,
                  const double* theta__ring,
                  const double support_rad,
                  const double conversion_factor,
                  const double delta_angle,
                  const float* kernel__alpha,
                  const int support_rings,
                  const int nr_interp_points,
                  const int nrings,
                  const int nside,
                  const int npix_ring,
                  const int nfftpix_ring) {

  cufftReal*                                                map_inout_gpu__pix;
  cufftComplex*                                             fftmap_in_gpu__pix;
  cufftComplex*                                             fftmap_out_gpu__pix;
  cufftHandle                                               forward_plan_kernel_gpu;
  cufftComplex*                                             fftkernel_gpu__pix;
  float*                                                    kernel_gpu__pix;
  size_t                                                    size_map_inout;
  size_t                                                    size_fftmap_inout;
  size_t                                                    size_kernel;
  size_t                                                    size_fftkernel;
  size_t                                                    size_interp_kernel;
  size_t                                                    size_ntheta;

  checkCudaErrors(cudaFuncSetCacheConfig(kernel2grid_gpu_device,
                                         cudaFuncCachePreferL1));
  checkCudaErrors(cudaFuncSetCacheConfig(rotate_fftrings_device,
                                         cudaFuncCachePreferL1));
  checkCudaErrors(cudaFuncSetCacheConfig(kernel_times_map_gpu_device,
                                         cudaFuncCachePreferL1));

  size_map_inout     = sizeof(cufftReal)*nrings*npix_ring;
  size_fftmap_inout  = sizeof(cufftComplex)*nrings*nfftpix_ring;
  size_kernel        = sizeof(cufftReal)*(2*support_rings+1)*npix_ring;
  size_fftkernel     = sizeof(cufftComplex)*(2*support_rings+1)*nfftpix_ring;
  size_interp_kernel = sizeof(float)*nr_interp_points;
  size_ntheta        = sizeof(double)*2*nside;

  checkCudaErrors(cudaMalloc((void**) &map_inout_gpu__pix, size_map_inout));
  checkCudaErrors(cudaMalloc((void**) &fftmap_in_gpu__pix, size_fftmap_inout));
  checkCudaErrors(cudaMalloc((void**) &fftmap_out_gpu__pix, size_fftmap_inout));
  checkCudaErrors(cudaMemset(map_inout_gpu__pix, 0, size_map_inout));
  checkCudaErrors(cudaMemset(fftmap_in_gpu__pix, 0, size_fftmap_inout));
  checkCudaErrors(cudaMemset(fftmap_out_gpu__pix, 0, size_fftmap_inout));

  checkCudaErrors(cudaMemcpy(map_inout_gpu__pix, map_in__pix, size_map_inout,
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((void**) &kernel_gpu__pix, size_kernel));
  checkCudaErrors(cudaMalloc((void**) &fftkernel_gpu__pix, size_fftkernel));
  checkCudaErrors(cudaMemset(kernel_gpu__pix, 0, size_kernel));
  checkCudaErrors(cudaMemset(fftkernel_gpu__pix, 0, size_fftkernel));

  checkCudaErrors(cudaMemcpyToSymbol(kernel_gpu__alpha, kernel__alpha,
                                     size_interp_kernel));

  checkCudaErrors(cudaMemcpyToSymbol(theta_gpu__ring, theta__ring, size_ntheta));

  fft_map_gpu(fftmap_in_gpu__pix, map_inout_gpu__pix, nside, nrings,
              npix_ring, nfftpix_ring);

  checkCudaErrors(cufftPlan1d(&forward_plan_kernel_gpu, npix_ring, CUFFT_R2C,
                              2*support_rings+1));

  convolve_with_kernel_gpu(fftmap_out_gpu__pix, fftmap_in_gpu__pix,
                           fftkernel_gpu__pix, kernel_gpu__pix,
                           forward_plan_kernel_gpu, conversion_factor,
                           delta_angle, support_rad,
                           support_rings, nside, nrings,
                           npix_ring, nfftpix_ring);

  ifft_map_gpu(map_inout_gpu__pix, fftmap_out_gpu__pix,
               nside, nrings, npix_ring, nfftpix_ring);

  checkCudaErrors(cudaMemcpy(map_out__pix, map_inout_gpu__pix, size_map_inout,
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(map_inout_gpu__pix));
  checkCudaErrors(cudaFree(fftmap_in_gpu__pix));
  checkCudaErrors(cudaFree(fftmap_out_gpu__pix));
  checkCudaErrors(cudaFree(kernel_gpu__pix));
  checkCudaErrors(cudaFree(fftkernel_gpu__pix));
  checkCudaErrors(cufftDestroy(forward_plan_kernel_gpu));

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// FFT map
//-----------------------------------------------------------------------

void fft_map_gpu(cufftComplex* fftmap_in_gpu__pix,
                 cufftReal* map_in_gpu__pix, const int nside,
                 const int nrings, const int npix_ring,
                 const int nfftpix_ring) {

  int                                                       npix;
  cufftHandle                                               forward_plan;
  dim3                                                      blocks_rot;
  dim3                                                      threads_rot;

  blocks_rot.x  = nside/32;
  blocks_rot.y  = nside+1;
  blocks_rot.z  = 1;
  threads_rot.x = 64;
  threads_rot.y =  2;
  threads_rot.z =  1;

  for (int i=0; i<nside-1; i++) {

    npix = 4*(i+1);

    checkCudaErrors(cufftPlan1d(&forward_plan, npix, CUFFT_R2C, 1));

    checkCudaErrors(cufftExecR2C(forward_plan, &map_in_gpu__pix[npix_ring*i],
                                 &fftmap_in_gpu__pix[nfftpix_ring*i]));

    checkCudaErrors(cufftExecR2C(forward_plan,
                                 &map_in_gpu__pix[npix_ring*(nrings-1-i)],
                                 &fftmap_in_gpu__pix[nfftpix_ring*(nrings-1-i)]));

    checkCudaErrors(cufftDestroy(forward_plan));

  };

  if (nside < 2048) {

    checkCudaErrors(cufftPlan1d(&forward_plan, npix_ring, CUFFT_R2C, 2*nside+1));

    checkCudaErrors(cufftExecR2C(forward_plan,
                                 &map_in_gpu__pix[npix_ring*(nside-1)],
                                 &fftmap_in_gpu__pix[nfftpix_ring*(nside-1)]));

    checkCudaErrors(cufftDestroy(forward_plan));

  } else {

    checkCudaErrors(cufftPlan1d(&forward_plan, npix_ring, CUFFT_R2C, nside+1));

    checkCudaErrors(cufftExecR2C(forward_plan,
                                 &map_in_gpu__pix[npix_ring*(nside-1)],
                                 &fftmap_in_gpu__pix[nfftpix_ring*(nside-1)]));

    checkCudaErrors(cufftDestroy(forward_plan));

    checkCudaErrors(cufftPlan1d(&forward_plan, npix_ring, CUFFT_R2C, nside));

    checkCudaErrors(cufftExecR2C(forward_plan,
                                 &map_in_gpu__pix[npix_ring*(2*nside)],
                                 &fftmap_in_gpu__pix[nfftpix_ring*(2*nside)]));

    checkCudaErrors(cufftDestroy(forward_plan));

  };

  rotate_fftrings_device<<<blocks_rot, threads_rot>>>
    (fftmap_in_gpu__pix, -1.0f, float (npix_ring), nfftpix_ring,
     nrings, nside);

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Shift FFT coefficients
//-----------------------------------------------------------------------

__global__ void rotate_fftrings_device(cufftComplex* fftmap_gpu__pix,
                                       const float direction,
                                       const float npix_ring_f,
                                       const int nfftpix_ring,
                                       const int nrings,
                                       const int nside) {

  int                                                       ring_pix;
  int                                                       ring;
  float                                                     shiftangle;
  float                                                     sinshift;
  float                                                     cosshift;
  cufftComplex                                              temp;

  if (threadIdx.y == 0) {

    ring     = blockIdx.y;
    ring_pix = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if ((ring_pix < 2*(ring+1)+1) && (ring < nside-1)) {

      shiftangle = direction * PI_f/(float (4*(ring+1))) * (float (ring_pix));
      __sincosf(shiftangle, &sinshift, &cosshift);
      temp.x = cosshift;
      temp.y = sinshift;

      fftmap_gpu__pix[nfftpix_ring*ring+ring_pix]
        = cuCmulf(fftmap_gpu__pix[nfftpix_ring*ring+ring_pix], temp);

      fftmap_gpu__pix[nfftpix_ring*(nrings-1-ring)+ring_pix]
        = cuCmulf(fftmap_gpu__pix[nfftpix_ring*(nrings-1-ring)+ring_pix], temp);

    };

  } else {

    ring     = 2*blockIdx.y + nside-1;
    ring_pix = blockIdx.x * blockDim.x + threadIdx.x + 1;

    shiftangle = direction * PI_f/npix_ring_f * (float (ring_pix));
    __sincosf(shiftangle, &sinshift, &cosshift);
    temp.x = cosshift;
    temp.y = sinshift;

    fftmap_gpu__pix[nfftpix_ring*ring+ring_pix]
      = cuCmulf(fftmap_gpu__pix[nfftpix_ring*ring+ring_pix], temp);

  };

};

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Do convolution
//-----------------------------------------------------------------------

void convolve_with_kernel_gpu(cufftComplex* fftmap_out_gpu__pix,
                              cufftComplex* fftmap_in_gpu__pix,
                              cufftComplex* fftkernel_gpu__pix,
                              float* kernel_gpu__pix,
                              cufftHandle forward_plan_kernel_gpu,
                              const double conversion_factor,
                              const double delta_angle,
                              const float support_rad,
                              const int support_rings,
                              const int nside,
                              const int nrings,
                              const int npix_ring,
                              const int nfftpix_ring) {

  dim3                                                      blocks_real;
  dim3                                                      blocks_fft;
  dim3                                                      threads_real;
  dim3                                                      threads_fft;

  threads_real.x = 128;
  threads_real.y =   1;
  threads_real.z =   1;
  blocks_real.x  = npix_ring/(16*threads_real.x);
  blocks_real.y  = 2*support_rings+1;
  blocks_real.z  = 1;

  threads_fft.x = 64;
  threads_fft.y =  2;
  threads_fft.z =  1;
  blocks_fft.x  = int (ceilf((float (nfftpix_ring)) / (float (threads_fft.x))));
  blocks_fft.y  = int (ceilf((float (2*support_rings+1)) / 32.0f));
  blocks_fft.z  = 1;

  for (int i=0; i<nrings/2+1; i++) {

    if (i == nrings/2) threads_fft.y = 1;

    kernel2grid_gpu_device<<<blocks_real, threads_real>>>
      (kernel_gpu__pix, conversion_factor, delta_angle, support_rad,
       nside, nrings, npix_ring, support_rings, i);

    checkCudaErrors(cufftExecR2C(forward_plan_kernel_gpu, kernel_gpu__pix,
                                 fftkernel_gpu__pix));

    kernel_times_map_gpu_device<<<blocks_fft, threads_fft>>>
      (fftmap_out_gpu__pix, fftmap_in_gpu__pix, fftkernel_gpu__pix,
       i, nrings, support_rings, npix_ring, nfftpix_ring);

  };

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Generate kernel on grid
//-----------------------------------------------------------------------

__global__ void kernel2grid_gpu_device(float* kernel_gpu__pix,
                                       const double conversion_factor,
                                       const double delta_angle,
                                       const float support_rad,
                                       const int nside,
                                       const int nrings,
                                       const int npix_ring,
                                       const int support_rings,
                                       const int base_ring) {

  int                                                       ring_index;
  int                                                       ring;
  int                                                       ring_pix[8];
  double                                                    theta_ring_index;
  double                                                    sint1_sqr;
  double                                                    sint2_sqr;
  double                                                    anglesum;
  double                                                    angledif;
  double                                                    angle_sqr[8];

  ring = blockIdx.y;

  ring_pix[0] = 8*blockDim.x * blockIdx.x                + threadIdx.x + 1;
  ring_pix[1] = 8*blockDim.x * blockIdx.x +   blockDim.x + threadIdx.x + 1;
  ring_pix[2] = 8*blockDim.x * blockIdx.x + 2*blockDim.x + threadIdx.x + 1;
  ring_pix[3] = 8*blockDim.x * blockIdx.x + 3*blockDim.x + threadIdx.x + 1;
  ring_pix[4] = 8*blockDim.x * blockIdx.x + 4*blockDim.x + threadIdx.x + 1;
  ring_pix[5] = 8*blockDim.x * blockIdx.x + 5*blockDim.x + threadIdx.x + 1;
  ring_pix[6] = 8*blockDim.x * blockIdx.x + 6*blockDim.x + threadIdx.x + 1;
  ring_pix[7] = 8*blockDim.x * blockIdx.x + 7*blockDim.x + threadIdx.x + 1;

  ring_index = base_ring + ring - support_rings;

  if ((ring_index < 0) || (ring_index > nrings-1)) return;

  if (ring_index < 2*nside) {
    theta_ring_index = theta_gpu__ring[ring_index];
  } else {
    theta_ring_index = PI_d - theta_gpu__ring[nrings-1-ring_index];
  };

  sint1_sqr = sin(0.5*(theta_ring_index + theta_gpu__ring[base_ring]));
  sint2_sqr = sin(0.5*(theta_ring_index - theta_gpu__ring[base_ring]));
  sint1_sqr = sint1_sqr * sint1_sqr;
  sint2_sqr = sint2_sqr * sint2_sqr;
  anglesum  = 2.0*(sint1_sqr + sint2_sqr);
  angledif  = 2.0*(sint1_sqr - sint2_sqr);

  angle_sqr[0] = anglesum - angledif * cos((double (ring_pix[0])) * delta_angle);
  angle_sqr[1] = anglesum - angledif * cos((double (ring_pix[1])) * delta_angle);
  angle_sqr[2] = anglesum - angledif * cos((double (ring_pix[2])) * delta_angle);
  angle_sqr[3] = anglesum - angledif * cos((double (ring_pix[3])) * delta_angle);
  angle_sqr[4] = anglesum - angledif * cos((double (ring_pix[4])) * delta_angle);
  angle_sqr[5] = anglesum - angledif * cos((double (ring_pix[5])) * delta_angle);
  angle_sqr[6] = anglesum - angledif * cos((double (ring_pix[6])) * delta_angle);
  angle_sqr[7] = anglesum - angledif * cos((double (ring_pix[7])) * delta_angle);

  kernel_gpu__pix[npix_ring*ring+ring_pix[0]]
    = interpolate_gpu_device(angle_sqr[0], support_rad, conversion_factor);
  kernel_gpu__pix[npix_ring*ring+ring_pix[1]]
    = interpolate_gpu_device(angle_sqr[1], support_rad, conversion_factor);
  kernel_gpu__pix[npix_ring*ring+ring_pix[2]]
    = interpolate_gpu_device(angle_sqr[2], support_rad, conversion_factor);
  kernel_gpu__pix[npix_ring*ring+ring_pix[3]]
    = interpolate_gpu_device(angle_sqr[3], support_rad, conversion_factor);
  kernel_gpu__pix[npix_ring*ring+ring_pix[4]]
    = interpolate_gpu_device(angle_sqr[4], support_rad, conversion_factor);
  kernel_gpu__pix[npix_ring*ring+ring_pix[5]]
    = interpolate_gpu_device(angle_sqr[5], support_rad, conversion_factor);
  kernel_gpu__pix[npix_ring*ring+ring_pix[6]]
    = interpolate_gpu_device(angle_sqr[6], support_rad, conversion_factor);
  kernel_gpu__pix[npix_ring*ring+ring_pix[7]]
    = interpolate_gpu_device(angle_sqr[7], support_rad, conversion_factor);

  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[0]]
    = kernel_gpu__pix[npix_ring*ring+ring_pix[0]];
  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[1]]
    = kernel_gpu__pix[npix_ring*ring+ring_pix[1]];
  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[2]]
    = kernel_gpu__pix[npix_ring*ring+ring_pix[2]];
  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[3]]
    = kernel_gpu__pix[npix_ring*ring+ring_pix[3]];
  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[4]]
    = kernel_gpu__pix[npix_ring*ring+ring_pix[4]];
  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[5]]
    = kernel_gpu__pix[npix_ring*ring+ring_pix[5]];
  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[6]]
    = kernel_gpu__pix[npix_ring*ring+ring_pix[6]];
  kernel_gpu__pix[npix_ring*(ring+1)-ring_pix[7]]
    = kernel_gpu__pix[npix_ring*ring+ring_pix[7]];

  if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
    angle_sqr[0] = anglesum - angledif;
    kernel_gpu__pix[npix_ring*ring]
      = interpolate_gpu_device(angle_sqr[0], support_rad, conversion_factor);
  };

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Interpolate kernel
//-----------------------------------------------------------------------

__device__  __inline__ float interpolate_gpu_device(double angle,
                                                    const double support_rad,
                                                    const double conversion_factor) {

  if (angle >= support_rad) return 0.0f;

  int                                                       lo;
  int                                                       hi;
  float                                                     fractional_part;
  double                                                    integer_part;

  fractional_part = float (modf(conversion_factor * angle, &integer_part));

  lo = int (integer_part);
  hi = lo + 1;

  return kernel_gpu__alpha[lo]
    + (kernel_gpu__alpha[hi] - kernel_gpu__alpha[lo]) * fractional_part;

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Multiply kernel and map
//-----------------------------------------------------------------------

__global__ void kernel_times_map_gpu_device(cufftComplex* fftmap_out_gpu__pix,
                                            const cufftComplex* fftmap_in_gpu__pix,
                                            const cufftComplex* fftkernel_gpu__pix,
                                            const int base_ring,
                                            const int nrings,
                                            const int support_rings,
                                            const int npix_ring,
                                            const int nfftpix_ring) {

  int                                                       ring_pix;
  int                                                       ring_index;
  int                                                       ring;
  cufftComplex                                              temp;

  if (threadIdx.y == 0) {

    ring     = blockIdx.y * 32;
    ring_pix = blockIdx.x * blockDim.x + threadIdx.x;

    ring_index = base_ring + ring - support_rings;

    if (ring_pix > min(nfftpix_ring-1, 2*(base_ring+1))) return;

    temp.x = 0.0f;
    temp.y = 0.0f;

    for (int i=max(0, -ring_index); i<min(32, min(nrings-ring_index,
                                          2*support_rings+1-ring)); i++) {

      temp = cuCaddf(cuCmulf(fftkernel_gpu__pix[nfftpix_ring*(ring+i)+ring_pix],
              fftmap_in_gpu__pix[nfftpix_ring*(ring_index+i)+ring_pix]), temp);

    };

    atomicAdd(&fftmap_out_gpu__pix[nfftpix_ring*base_ring+ring_pix].x, temp.x);
    atomicAdd(&fftmap_out_gpu__pix[nfftpix_ring*base_ring+ring_pix].y, temp.y);

  } else {

    ring       = 2*support_rings - blockIdx.y * 32;
    ring_pix   = blockIdx.x * blockDim.x + threadIdx.x;

    ring_index = nrings-1 - base_ring - ring + support_rings;

    if (ring_pix > min(nfftpix_ring-1, 2*(base_ring+1))) return;

    temp.x = 0.0f;
    temp.y = 0.0f;

    for (int i=max(0, -ring_index); i<min(32, min(nrings-ring_index,
                                                  ring+1)); i++) {

      temp = cuCaddf(cuCmulf(fftkernel_gpu__pix[nfftpix_ring*(ring-i)+ring_pix],
                fftmap_in_gpu__pix[nfftpix_ring*(ring_index+i)+ring_pix]), temp);

    };

    atomicAdd(&fftmap_out_gpu__pix[nfftpix_ring*(nrings-1-base_ring)+ring_pix].x,
              temp.x);
    atomicAdd(&fftmap_out_gpu__pix[nfftpix_ring*(nrings-1-base_ring)+ring_pix].y,
              temp.y);

  };

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Inverse FFT map
//-----------------------------------------------------------------------

void ifft_map_gpu(cufftReal* map_out_gpu__pix,
                  cufftComplex* fftmap_out_gpu__pix,
                  const int nside, const int nrings,
                  const int npix_ring, const int nfftpix_ring) {

  int                                                       npix;
  cufftHandle                                               backward_plan;
  dim3                                                      blocks_rot;
  dim3                                                      threads_rot;

  blocks_rot.x  = nside/32;
  blocks_rot.y  = nside+1;
  blocks_rot.z  = 1;
  threads_rot.x = 64;
  threads_rot.y =  2;
  threads_rot.z =  1;

  rotate_fftrings_device<<<blocks_rot, threads_rot>>>
    (fftmap_out_gpu__pix, 1.0f, float (npix_ring), nfftpix_ring,
     nrings, nside);

  for (int i=0; i<nside-1; i++) {

    npix = 2*(i+1);
    checkCudaErrors(cudaMemset(&fftmap_out_gpu__pix[nfftpix_ring*i+npix].y,
                               0, sizeof(cufftReal)));
    checkCudaErrors(cudaMemset(&fftmap_out_gpu__pix[nfftpix_ring*(nrings-1-i)+npix].y,
                               0, sizeof(cufftReal)));

    npix = 4*(i+1);

    checkCudaErrors(cufftPlan1d(&backward_plan, npix, CUFFT_C2R, 1));

    checkCudaErrors(cufftExecC2R(backward_plan,
                                 &fftmap_out_gpu__pix[nfftpix_ring*i],
                                 &map_out_gpu__pix[npix_ring*i]));

    checkCudaErrors(cufftExecC2R(backward_plan,
                                 &fftmap_out_gpu__pix[nfftpix_ring*(nrings-1-i)],
                                 &map_out_gpu__pix[npix_ring*(nrings-1-i)]));

    checkCudaErrors(cufftDestroy(backward_plan));

  };

  if (nside < 2048) {

    checkCudaErrors(cufftPlan1d(&backward_plan, npix_ring, CUFFT_C2R, 2*nside+1));

    checkCudaErrors(cufftExecC2R(backward_plan,
                                 &fftmap_out_gpu__pix[nfftpix_ring*(nside-1)],
                                 &map_out_gpu__pix[npix_ring*(nside-1)]));

    checkCudaErrors(cufftDestroy(backward_plan));

  } else {

    checkCudaErrors(cufftPlan1d(&backward_plan, npix_ring, CUFFT_C2R, nside+1));

    checkCudaErrors(cufftExecC2R(backward_plan,
                                 &fftmap_out_gpu__pix[nfftpix_ring*(nside-1)],
                                 &map_out_gpu__pix[npix_ring*(nside-1)]));

    checkCudaErrors(cufftDestroy(backward_plan));

    checkCudaErrors(cufftPlan1d(&backward_plan, npix_ring, CUFFT_C2R, nside));

    checkCudaErrors(cufftExecC2R(backward_plan,
                                 &fftmap_out_gpu__pix[nfftpix_ring*(2*nside)],
                                 &map_out_gpu__pix[npix_ring*(2*nside)]));

    checkCudaErrors(cufftDestroy(backward_plan));

  };

}

//-----------------------------------------------------------------------
