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

//-----------------------------------------------------------------------
// Auxiliary data for spherical harmonic transforms
//-----------------------------------------------------------------------

class sht_aux {

//-----------------------------------------------------------------------

  friend class skymap;
  friend class convmap;

 private:

  int                                                       stride_pix;
  int                                                       stride_alm;

  double*                                                   w8rings;
  psht_alm_info*                                            alm_info;
  psht_geom_info*                                           geom_info;

 protected:

 public:

  sht_aux();

 ~sht_aux();

  void init(const int, const int, const int);

  void free();

};

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Collection of basic simulation parameters
//-----------------------------------------------------------------------

class parameter {

//-----------------------------------------------------------------------

 private:

 protected:

 public:

  bool                                                      do_gpu;
  bool                                                      pol;
  int                                                       nside;
  int                                                       nrings;
  int                                                       lmax;
  int                                                       mmax;
  int                                                       npix_ring;
  int                                                       nfftpix_ring;
  int                                                       npix;
  int                                                       nalm;
  sht_aux                                                   sht;

  parameter();

 ~parameter();

  void init(int, int, int, bool);

  void free();

  void query();

};

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Map in pixel and harmonic space T only or T & E & B
//-----------------------------------------------------------------------

class skymap {

//-----------------------------------------------------------------------

 private:

 protected:

  bool                                                      pol;
  int                                                       nside;
  int                                                       lmax;
  int                                                       mmax;
  int                                                       npix;
  int                                                       nalm;

 public:

  float*                                                    map_I;
  float*                                                    map_Q;
  float*                                                    map_U;
  pshts_cmplx*                                              alm_T;
  pshts_cmplx*                                              alm_E;
  pshts_cmplx*                                              alm_B;

  skymap();

 ~skymap();

  void init(parameter);

  void free();

  void fits2map(const std::string&, const parameter&);

  void map2fits(const std::string&, const parameter&);

  void map2alm(const parameter&);

  void alm2map(const parameter&);

  void query();

};

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Radially symmetric convolution kernel
//-----------------------------------------------------------------------

class convkernel {

//-----------------------------------------------------------------------

  friend class convmap;

 private:

  std::vector<double>                                       vec_angdist__alpha;
  std::vector<double>                                       vec_kernel__alpha;

  fftwf_plan                                                forward_plan_kernel;
  int                                                       nr_kernel_nodes;
  int                                                       support_rings;
  float                                                     delta_angle;
  float*                                                    angdist__alpha;
  float*                                                    kernel__alpha;
  float*                                                    kernel__pix;
  double                                                    support_rad;
  double                                                    conversion_factor;
  double*                                                   theta__ring;
  double*                                                   shift__ring;
  double*                                                   phi__ringpix;
  fftwf_complex*                                            fftkernel__pix;

  void allocate_fft_kernel(const parameter);

  void generate_kernel_grid(const parameter);

  void kernel2grid(const int, const parameter);

  void kernel2grid_shift(const int, const parameter);

  float interp_kernel(const double&);

  void fft_kernel(const parameter);

  void rotate_kernel(const int, const parameter);

 protected:

 public:

  convkernel();

 ~convkernel();

  void init(const std::string&, parameter&);

  void free();

};

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Map with additional parameters for convolution
//-----------------------------------------------------------------------

class convmap : public skymap {

//-----------------------------------------------------------------------

 private:

  fftwf_plan*                                               forward_plan_map;
  fftwf_plan*                                               backward_plan_map;
  float*                                                    map_in__pix;
  float*                                                    map_out__pix;
  fftwf_complex*                                            fftmap_in__pix;
  fftwf_complex*                                            fftmap_out__pix;

  void allocate_fft_map(convkernel, const parameter);

  void allocate_fft_map_gpu(const convkernel, const parameter);

  void map2grid(const skymap, const convkernel, const parameter);

  void fft_map(const convkernel, const parameter);

  void rotate_fftrings(fftwf_complex*, const parameter, const float);

  void convolve_with_kernel(convkernel, const parameter);

  void convolve_with_kernel_shift(convkernel, const parameter);

  void ifft_map(const convkernel, const parameter);

  void grid2map(const convkernel, const parameter);

 protected:

 public:

  convmap();

 ~convmap();

  void init(parameter);

  void free();

  void convolve(const skymap, convkernel, const parameter);

  void get_first_ring(const skymap, convkernel, const parameter);

};

//-----------------------------------------------------------------------
