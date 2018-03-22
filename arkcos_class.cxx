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

#include "arkcos_main.hxx"

using namespace std;

//-----------------------------------------------------------------------
// Member functions: sht_aux
//-----------------------------------------------------------------------

sht_aux::sht_aux() {

  stride_pix = 0;
  stride_alm = 0;
  w8rings    = NULL;
  alm_info   = NULL;
  geom_info  = NULL;

}

sht_aux::~sht_aux() {

}

void sht_aux::init(const int nside, const int lmax, const int mmax) {

  stride_pix = 1;
  stride_alm = 1;

  w8rings = new double [2*nside];
  for (int i=0; i<2*nside; i++) {
    w8rings[i] = 1.0;
  };

  psht_make_weighted_healpix_geom_info(nside, stride_pix,
                                       w8rings, &geom_info);

  psht_make_rectangular_alm_info(lmax, mmax, stride_alm, &alm_info);

}

void sht_aux::free() {

  delete [] w8rings;
  w8rings = NULL;

  psht_destroy_alm_info(alm_info);
  psht_destroy_geom_info(geom_info);
  alm_info  = NULL;
  geom_info = NULL;

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Member functions: parameter
//-----------------------------------------------------------------------

parameter::parameter() {

  do_gpu       = false;
  pol          = false;
  nside        = 0;
  nrings       = 0;
  npix_ring    = 0;
  nfftpix_ring = 0;
  npix         = 0;
  nalm         = 0;
  lmax         = 0;
  mmax         = 0;

}

parameter::~parameter() {

}

void parameter::init(const int nside_, const int lmax_,
                     const int mmax_, const bool pol_) {

  do_gpu       = false;
  pol          = pol_;
  nside        = nside_;
  nrings       = 4*nside-1;
  npix_ring    = 4*nside;
  nfftpix_ring = 2*nside+1;
  npix         = 12*nside*nside;
  lmax         = lmax_;
  mmax         = mmax_;
  nalm         = (lmax+1)*(mmax+1);

  sht.init(nside, lmax, mmax);

}

void parameter::free() {

  sht.free();

}

void parameter::query() {

  cout << "Parameter:"         << endl;
  cout << " nside = " << nside << endl;
  cout << " lmax  = " << lmax  << endl;
  cout << " mmax  = " << mmax  << endl;
  if (pol) {
    cout << " pol   = true"    << endl;
  } else {
    cout << " pol   = false"   << endl;
  };
  if (do_gpu) {
    cout << " GPU   = true"    << endl;
  } else {
    cout << " GPU   = false"   << endl;
  };

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Member functions: skymap
//-----------------------------------------------------------------------

skymap::skymap() {

  pol   = false;
  nside = 0;
  npix  = 0;
  nalm  = 0;
  lmax  = 0;
  mmax  = 0;
  map_I = NULL;
  map_Q = NULL;
  map_U = NULL;
  alm_T = NULL;
  alm_E = NULL;
  alm_B = NULL;

}

skymap::~skymap() {

}

void skymap::init(const parameter par) {

  pol   = par.pol;
  nside = par.nside;
  npix  = par.npix;
  nalm  = par.nalm;
  lmax  = par.lmax;
  mmax  = par.mmax;

  map_I = new float [npix];
  alm_T = new pshts_cmplx [nalm];

  if (pol) {
    map_Q = new float [npix];
    map_U = new float [npix];
    alm_E = new pshts_cmplx [nalm];
    alm_B = new pshts_cmplx [nalm];
  };

}

void skymap::free() {

  delete [] map_I;
  delete [] alm_T;
  map_I = NULL;
  alm_T = NULL;

  if (pol) {
    delete [] map_Q;
    delete [] map_U;
    delete [] alm_E;
    delete [] alm_B;
    map_Q = NULL;
    map_U = NULL;
    alm_E = NULL;
    alm_B = NULL;
  };

}

void skymap::map2alm(const parameter& par) {

  pshts_joblist*                                            joblist;

  pshts_make_joblist(&joblist);

  if (pol) {
    pshts_add_job_map2alm_pol(joblist, map_I, map_Q, map_U,
                              alm_T, alm_E, alm_B, 0);
  } else {
    pshts_add_job_map2alm(joblist, map_I, alm_T, 0);
  };

  pshts_execute_jobs(joblist, par.sht.geom_info, par.sht.alm_info);

  pshts_destroy_joblist(joblist);

}

void skymap::alm2map(const parameter& par) {

  pshts_joblist*                                            joblist;

  pshts_make_joblist(&joblist);

  if (pol) {
    pshts_add_job_alm2map_pol(joblist, alm_T, alm_E, alm_B,
                              map_I, map_Q, map_U, 0);
  } else {
    pshts_add_job_alm2map(joblist, alm_T, map_I, 0);
  };

  pshts_execute_jobs(joblist, par.sht.geom_info, par.sht.alm_info);

  pshts_destroy_joblist(joblist);

}

void skymap::fits2map(const string& infile, const parameter& par) {

  fitshandle filehandle;
  Healpix_Map<float> hmap(par.nside, RING, SET_NSIDE);

  filehandle.open(infile);
  filehandle.goto_hdu(2);

  read_Healpix_map_from_fits(filehandle, hmap, 1);
  for (int i=0;i<par.npix;i++) {
    map_I[i] = hmap[i];
  };

  if (par.pol) {
    read_Healpix_map_from_fits(infile, hmap, 2, 2);
    for (int i=0;i<par.npix;i++) {
      map_Q[i] = hmap[i];
    };
    read_Healpix_map_from_fits(infile, hmap, 3, 2);
    for (int i=0;i<par.npix;i++) {
      map_U[i] = hmap[i];
    };
  };

}

void skymap::map2fits(const string& outfile, const parameter& par) {

  fitshandle filehandle;
  filehandle.create(outfile);

  if (par.pol) {
    Healpix_Map<float> hmap_I(par.nside, RING, SET_NSIDE);
    Healpix_Map<float> hmap_Q(par.nside, RING, SET_NSIDE);
    Healpix_Map<float> hmap_U(par.nside, RING, SET_NSIDE);
    for (int i=0;i<par.npix;i++) {
      hmap_I[i] = map_I[i];
      hmap_Q[i] = map_Q[i];
      hmap_U[i] = map_U[i];
    };
    write_Healpix_map_to_fits(filehandle, hmap_I, hmap_Q, hmap_U, PLANCK_FLOAT32);
  } else {
    Healpix_Map<float> hmap_I(par.nside, RING, SET_NSIDE);
    for (int i=0;i<par.npix;i++) {
      hmap_I[i] = map_I[i];
    };
    write_Healpix_map_to_fits(filehandle, hmap_I, PLANCK_FLOAT32);
  };

}

void skymap::query() {

  cout << "Map:"               << endl;
  cout << " nside = " << nside << endl;
  cout << " lmax  = " << lmax  << endl;
  cout << " mmax  = " << mmax  << endl;
  if (pol) {
    cout << " pol   = true"    << endl;
  } else {
    cout << " pol   = false"   << endl;
  };

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Member functions: convkernel
//-----------------------------------------------------------------------

convkernel::convkernel() {

  nr_kernel_nodes   = 0;
  support_rings     = 0;
  delta_angle       = 0.0f;
  support_rad       = 0.0;
  conversion_factor = 0.0;
  angdist__alpha    = NULL;
  kernel__alpha     = NULL;
  kernel__pix       = NULL;
  theta__ring       = NULL;
  shift__ring       = NULL;
  phi__ringpix      = NULL;
  fftkernel__pix    = NULL;
  vec_angdist__alpha.clear();
  vec_kernel__alpha.clear();

}

convkernel::~convkernel() {

}

void convkernel::init(const string& infile, parameter& par) {

  int                                                       nr_interp_points;
  double                                                    step;
  double                                                    read_theta;
  double                                                    read_kernel;
  double                                                    largest_element;
  double                                                    support_threshold;
  double                                                    deviation;
  double*                                                   first_derivative_file;
  double*                                                   angdist_file__alpha;
  double*                                                   kernel_file__alpha;
  double*                                                   angdist_interp__alpha;
  double*                                                   kernel_interp__alpha;

  ifstream input_file(infile.c_str());

  if (input_file.fail()) {
    cout << "Error: Could not open kernel file" << endl;
    abort();
  };

  while (true) {
    input_file >> read_theta >> read_kernel;
    if ((input_file.eof()) or (read_theta > PI_d)) break;
    vec_angdist__alpha.push_back(read_theta);
    vec_kernel__alpha.push_back(read_kernel);
  };

  input_file.close();

  largest_element
    = abs(*max_element(vec_kernel__alpha.begin(), vec_kernel__alpha.end()));
  support_threshold = 1.0E-5 * largest_element;

  for (int i=vec_angdist__alpha.size()-1; i>=0; i--) {
    support_rad = vec_angdist__alpha[i];
    if (abs(vec_kernel__alpha[i-1]) > support_threshold) break;
    if (int(vec_angdist__alpha.size())-2 <= i) continue;
    vec_angdist__alpha.pop_back();
    vec_kernel__alpha.pop_back();
  };
  support_rings = ceil((support_rad * double (par.nside))/0.66);
  support_rings = min(support_rings, par.nrings-1);

  nr_interp_points      = vec_angdist__alpha.size();
  first_derivative_file = new double [nr_interp_points];
  angdist_file__alpha   = new double [nr_interp_points];
  kernel_file__alpha    = new double [nr_interp_points];

  for (int i=0; i<nr_interp_points; i++) {
    deviation = (1.0 - 0.5*pow(vec_angdist__alpha[i], 2))
      - cos(vec_angdist__alpha[i]);
    angdist_file__alpha[i] = pow(vec_angdist__alpha[i], 2) + 2.0*deviation;
    kernel_file__alpha[i]  = vec_kernel__alpha[i];
  };
  angdist_file__alpha[nr_interp_points-1] = pow(PI_d, 2);
  kernel_file__alpha[nr_interp_points-1]  = 0.0f;

  nr_kernel_nodes       = 4000;
  angdist__alpha        = new float [nr_kernel_nodes];
  kernel__alpha         = new float [nr_kernel_nodes];
  angdist_interp__alpha = new double [nr_kernel_nodes];
  kernel_interp__alpha  = new double [nr_kernel_nodes];

  support_rad = pow(support_rad, 2);
  conversion_factor = (double (nr_kernel_nodes-1)) / support_rad;

  step = 1.0/conversion_factor;
  for (int i=0; i<nr_kernel_nodes; i++) {
    angdist_interp__alpha[i] = step * (double (i));
  };

  spline_pchip_set(nr_interp_points, angdist_file__alpha,
                   kernel_file__alpha, first_derivative_file);
  spline_pchip_val(nr_interp_points, angdist_file__alpha,
                   kernel_file__alpha, first_derivative_file,
                   nr_kernel_nodes, angdist_interp__alpha,
                   kernel_interp__alpha);

  for (int i=0; i<nr_kernel_nodes; i++) {
    angdist__alpha[i] = float (angdist_interp__alpha[i]);
    kernel__alpha[i]  = float (kernel_interp__alpha[i]);
  };

  generate_kernel_grid(par);

  allocate_fft_kernel(par);

  delete [] first_derivative_file;
  delete [] angdist_file__alpha;
  delete [] kernel_file__alpha;
  delete [] angdist_interp__alpha;
  delete [] kernel_interp__alpha;

}

void convkernel::generate_kernel_grid(const parameter par) {

  theta__ring  = new double [par.nrings];
  shift__ring  = new double [par.nrings];
  phi__ringpix = new double [par.npix_ring];
  memset(shift__ring, 0, sizeof(shift__ring)*par.nrings);

  delta_angle  = 2.0*PI_d/(double (par.npix_ring));

  for (int i=0; i<par.nrings; i++) {
    theta__ring[i] = ring2theta_double(i, par.nside, par.nrings);
  };

  for (int i=0; i<par.nside-1; i++) {
    shift__ring[i]              = PI_d/(double (4*(i+1)));
    shift__ring[par.nrings-1-i] = shift__ring[i];
  };

  for (int i=par.nside-1; i<3*par.nside; i+=2) {
    shift__ring[i] = 0.5*delta_angle;
  };

  for (int i=0; i<par.npix_ring; i++) {
    phi__ringpix[i] = (double (i)) * delta_angle;
  };

}

void convkernel::allocate_fft_kernel(const parameter par) {

  kernel__pix    = alloc_pseudo2d_arr_float(2*support_rings+1, par.npix_ring);
  fftkernel__pix = alloc_pseudo2d_arr_fftwf(2*support_rings+1, par.nfftpix_ring);

  forward_plan_kernel
    = fftwf_plan_dft_r2c_1d(par.npix_ring, kernel__pix,
                            fftkernel__pix, FFTW_MEASURE);

}

void convkernel::kernel2grid(const int base_ring, const parameter par) {

  int                                                       ring_index;
  float*                                                    output__pix;
  double                                                    sint2;
  double                                                    deltat2;
  double                                                    anglesum;
  double                                                    angledif;
  double                                                    angle2;

  output__pix = new float [par.npix_ring];

  for (int i=-support_rings; i<=support_rings; i++) {

    ring_index = base_ring + i;

    if ((0 > ring_index) || (par.nrings-1 < ring_index)) continue;

    sint2   = sin(0.5*(theta__ring[ring_index] + theta__ring[base_ring]));
    sint2   = sint2 * sint2;
    deltat2 = 0.5*(theta__ring[ring_index] - theta__ring[base_ring]);
    deltat2 = deltat2 * deltat2;

    anglesum = 2.0*(sint2 + deltat2);
    angledif = 2.0*(sint2 - deltat2);

    for (int j=0; j<par.npix_ring/2+1; j++) {
      angle2         = anglesum - angledif * cos(phi__ringpix[j]);
      output__pix[j] = interp_kernel(angle2);
    };

    for (int j=1; j<par.npix_ring/2; j++) {
      output__pix[par.npix_ring-j] = output__pix[j];
    };

    memcpy(&kernel__pix[par.npix_ring*(support_rings+i)], output__pix,
           sizeof(float)*par.npix_ring);

  };

  delete [] output__pix;

}

void convkernel::kernel2grid_shift(const int base_ring, const parameter par) {

  float*                                                    output__pix;
  double                                                    sint2;
  double                                                    deltat2;
  double                                                    anglesum;
  double                                                    angledif;
  double                                                    angle2;

  output__pix = new float [par.npix_ring];

  for (int i=max(base_ring-support_rings, 0);
       i<=min(base_ring+support_rings, par.nrings-1); i++) {

    sint2   = sin(0.5*(theta__ring[i] + theta__ring[base_ring]));
    sint2   = sint2 * sint2;
    deltat2 = 0.5*(theta__ring[i] - theta__ring[base_ring]);
    deltat2 = deltat2 * deltat2;

    anglesum = 2.0*(sint2 + deltat2);
    angledif = 2.0*(sint2 - deltat2);

    for (int j=0; j<par.npix_ring; j++) {
      angle2 = anglesum - angledif * cos(phi__ringpix[j] - shift__ring[i]);
      output__pix[j] = interp_kernel(angle2);
    };

    memcpy(&kernel__pix[par.npix_ring*(i-base_ring+support_rings)],
           output__pix, sizeof(float)*par.npix_ring);

  };

  delete [] output__pix;

}

float convkernel::interp_kernel(const double& angle) {

  if (angle >= support_rad) return 0.0f;

  int                                                       lo;
  int                                                       hi;
  float                                                     fractional_part;
  double                                                    integer_part;

  fractional_part = float (modf(conversion_factor * angle, &integer_part));

  lo = int (integer_part);
  hi = lo + 1;

  return kernel__alpha[lo]
    + (kernel__alpha[hi] - kernel__alpha[lo]) * fractional_part;

}

void convkernel::fft_kernel(const parameter par) {

  for (int i=0; i<2*support_rings+1; i++) {

    fftwf_execute_dft_r2c(forward_plan_kernel, &kernel__pix[par.npix_ring*i],
                          &fftkernel__pix[par.nfftpix_ring*i]);

  };

}

void convkernel::rotate_kernel(const int base_ring, const parameter par) {

  int                                                       ring;
  float                                                     shiftangle;
  float                                                     sinshift;
  float                                                     cosshift;
  float                                                     temp_real;
  float                                                     temp_imag;

  for (int i=0; i<2*support_rings+1; i++) {

    ring = base_ring - support_rings + i;

    if ((ring >= 0) && (ring <= par.nside-2)) {

      for (int j=1, k=1.0f; j<2*(ring+1)+1; j++, k+=1.0f) {

        shiftangle = -k*shift__ring[ring];
        sincosf(shiftangle, &sinshift, &cosshift);

        temp_real
          = cosshift * fftkernel__pix[par.nfftpix_ring*i+j][0]
          - sinshift * fftkernel__pix[par.nfftpix_ring*i+j][1];
        temp_imag
          = sinshift * fftkernel__pix[par.nfftpix_ring*i+j][0]
          + cosshift * fftkernel__pix[par.nfftpix_ring*i+j][1];
        fftkernel__pix[par.nfftpix_ring*i+j][0] = temp_real;
        fftkernel__pix[par.nfftpix_ring*i+j][1] = temp_imag;

      };

    } else if ((ring >= par.nside-1) && (ring <= 3*par.nside-1)) {

      for (int j=1, k=1.0f; j<par.nfftpix_ring; j++, k+=1.0f) {

        shiftangle = -k * shift__ring[ring];
        sincosf(shiftangle, &sinshift, &cosshift);

        temp_real
          = cosshift * fftkernel__pix[par.nfftpix_ring*i+j][0]
          - sinshift * fftkernel__pix[par.nfftpix_ring*i+j][1];
        temp_imag
          = sinshift * fftkernel__pix[par.nfftpix_ring*i+j][0]
          + cosshift * fftkernel__pix[par.nfftpix_ring*i+j][1];
        fftkernel__pix[par.nfftpix_ring*i+j][0] = temp_real;
        fftkernel__pix[par.nfftpix_ring*i+j][1] = temp_imag;

      };

    } else if ((ring >= 3*par.nside) && (ring <= par.nrings-1)) {

      for (int j=1, k=1.0f; j<2*(par.nrings-ring)+1; j++, k+=1.0f) {

        shiftangle = -k*shift__ring[ring];
        sincosf(shiftangle, &sinshift, &cosshift);

        temp_real
          = cosshift * fftkernel__pix[par.nfftpix_ring*i+j][0]
          - sinshift * fftkernel__pix[par.nfftpix_ring*i+j][1];
        temp_imag
          = sinshift * fftkernel__pix[par.nfftpix_ring*i+j][0]
          + cosshift * fftkernel__pix[par.nfftpix_ring*i+j][1];
        fftkernel__pix[par.nfftpix_ring*i+j][0] = temp_real;
        fftkernel__pix[par.nfftpix_ring*i+j][1] = temp_imag;

      };

    };

  };

}

void convkernel::free() {

  delete [] angdist__alpha;
  delete [] kernel__alpha;
  delete [] theta__ring;
  delete [] shift__ring;
  delete [] phi__ringpix;
  angdist__alpha = NULL;
  kernel__alpha  = NULL;
  theta__ring    = NULL;
  shift__ring    = NULL;
  phi__ringpix   = NULL;

  free_pseudo2d_arr_float(kernel__pix);
  free_pseudo2d_arr_fftwf(fftkernel__pix);
  kernel__pix    = NULL;
  fftkernel__pix = NULL;

  vec_angdist__alpha.clear();
  vec_kernel__alpha.clear();

  fftwf_destroy_plan(forward_plan_kernel);

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Member functions: convmap
//-----------------------------------------------------------------------

convmap::convmap() {

  pol             = false;
  nside           = 0;
  npix            = 0;
  nalm            = 0;
  lmax            = 0;
  mmax            = 0;
  map_I           = NULL;
  map_Q           = NULL;
  map_U           = NULL;
  alm_T           = NULL;
  alm_E           = NULL;
  alm_B           = NULL;
  map_in__pix     = NULL;
  map_out__pix    = NULL;
  fftmap_in__pix  = NULL;
  fftmap_out__pix = NULL;

}

convmap::~convmap() {

}

void convmap::init(const parameter par) {

  pol   = par.pol;
  nside = par.nside;
  npix  = par.npix;
  nalm  = par.nalm;
  lmax  = par.lmax;
  mmax  = par.mmax;

  map_I = new float [npix];
  alm_T = new pshts_cmplx [nalm];

  if (pol) {
    map_Q = new float [npix];
    map_U = new float [npix];
    alm_E = new pshts_cmplx [nalm];
    alm_B = new pshts_cmplx [nalm];
  };

}

void convmap::free() {

  int                                                       i;

  delete [] map_I;
  delete [] alm_T;
  map_I = NULL;
  alm_T = NULL;

  if (pol) {
    delete [] map_Q;
    delete [] map_U;
    delete [] alm_E;
    delete [] alm_B;
    map_Q = NULL;
    map_U = NULL;
    alm_E = NULL;
    alm_B = NULL;
  };

  free_pseudo2d_arr_float(map_in__pix);
  free_pseudo2d_arr_float(map_out__pix);
  free_pseudo2d_arr_fftwf(fftmap_in__pix);
  free_pseudo2d_arr_fftwf(fftmap_out__pix);
  map_in__pix     = NULL;
  map_out__pix    = NULL;
  fftmap_in__pix  = NULL;
  fftmap_out__pix = NULL;

  for (i=0; i<nside; i++) {
      fftwf_destroy_plan(forward_plan_map[i]);
      fftwf_destroy_plan(backward_plan_map[i]);
  }
  delete [] forward_plan_map;
  delete [] backward_plan_map;
  forward_plan_map  = NULL;
  backward_plan_map = NULL;

}

void convmap::allocate_fft_map(const convkernel kernel, const parameter par) {

  int                                                       i;

  map_in__pix     = alloc_pseudo2d_arr_float(par.nrings, par.npix_ring);
  map_out__pix    = alloc_pseudo2d_arr_float(par.nrings, par.npix_ring);
  fftmap_in__pix  = alloc_pseudo2d_arr_fftwf(par.nrings, par.nfftpix_ring);
  fftmap_out__pix = alloc_pseudo2d_arr_fftwf(par.nrings, par.nfftpix_ring);

  forward_plan_map  = new fftwf_plan [par.nside];
  backward_plan_map = new fftwf_plan [par.nside];

  for (i=0; i<par.nside-1; i++) {

    forward_plan_map[i]
      = fftwf_plan_dft_r2c_1d(4*(i+1), &map_in__pix[par.npix_ring*i],
                         &fftmap_in__pix[par.nfftpix_ring*i], FFTW_ESTIMATE);
    backward_plan_map[i]
      = fftwf_plan_dft_c2r_1d(4*(i+1), &fftmap_out__pix[par.nfftpix_ring*i],
                         &map_out__pix[par.npix_ring*i], FFTW_ESTIMATE);

  };

  i = par.nside-1;

  forward_plan_map[i]
    = fftwf_plan_dft_r2c_1d(par.npix_ring, &map_in__pix[par.npix_ring*i],
                            &fftmap_in__pix[par.nfftpix_ring*i], FFTW_MEASURE);
  backward_plan_map[i]
    = fftwf_plan_dft_c2r_1d(par.npix_ring, &fftmap_out__pix[par.nfftpix_ring*i],
                            &map_out__pix[par.npix_ring*i], FFTW_MEASURE);

}

void convmap::allocate_fft_map_gpu(convkernel kernel, const parameter par) {

  map_in__pix     = alloc_pseudo2d_arr_float(par.nrings, par.npix_ring);
  map_out__pix    = alloc_pseudo2d_arr_float(par.nrings, par.npix_ring);
  fftmap_in__pix  = alloc_pseudo2d_arr_fftwf(1, 1);
  fftmap_out__pix = alloc_pseudo2d_arr_fftwf(1, 1);
  forward_plan_map  = new fftwf_plan [1];
  backward_plan_map = new fftwf_plan [1];

  kernel.kernel__pix    = alloc_pseudo2d_arr_float(1, 1);
  kernel.fftkernel__pix = alloc_pseudo2d_arr_fftwf(1, 1);

}

void convmap::map2grid(const skymap map, const convkernel kernel,
                       const parameter par) {

  int                                                       npix_map_ring;
  int                                                       pix_offset;

  pix_offset    = 0;
  npix_map_ring = 0;

  for (int i=0; i<par.nside-1; i++) {

    npix_map_ring = 4*(i+1);

    for (int j=0; j<npix_map_ring; j++) {
      map_in__pix[par.npix_ring*i+j] = map.map_I[pix_offset+j];
    };

    pix_offset += npix_map_ring;

  };


  for (int i=par.nside-1; i<3*par.nside; i++) {

    memcpy(&map_in__pix[par.npix_ring*i], &map.map_I[pix_offset],
           sizeof(float)*par.npix_ring);

    pix_offset += par.npix_ring;

  };

  for (int i=3*par.nside; i<4*par.nside-1; i++) {

    npix_map_ring = 4*(4*par.nside-1 - i);

    for (int j=0; j<npix_map_ring; j++) {
      map_in__pix[par.npix_ring*i+j] = map.map_I[pix_offset+j];
    };

    pix_offset += npix_map_ring;

  };

}

void convmap::fft_map(const convkernel kernel, const parameter par) {

  for (int i=0; i<par.nside-1; i++) {

    fftwf_execute_dft_r2c(forward_plan_map[i], &map_in__pix[par.npix_ring*i],
                          &fftmap_in__pix[par.nfftpix_ring*i]);

    fftwf_execute_dft_r2c(forward_plan_map[i],
                          &map_in__pix[par.npix_ring*(par.nrings-1-i)],
                          &fftmap_in__pix[par.nfftpix_ring*(par.nrings-1-i)]);

    for (int j=2*(i+1)+1, h=2*(i+1)-1;
         j<min(4*(i+1), par.nfftpix_ring); j++, h--) {

      fftmap_in__pix[par.nfftpix_ring*i+j][0]
        = fftmap_in__pix[par.nfftpix_ring*i+h][0];
      fftmap_in__pix[par.nfftpix_ring*i+j][1]
        = -1.0f * fftmap_in__pix[par.nfftpix_ring*i+h][1];

      fftmap_in__pix[par.nfftpix_ring*(par.nrings-1-i)+j][0]
        = fftmap_in__pix[par.nfftpix_ring*(par.nrings-1-i)+h][0];
      fftmap_in__pix[par.nfftpix_ring*(par.nrings-1-i)+j][1]
        = -1.0f * fftmap_in__pix[par.nfftpix_ring*(par.nrings-1-i)+h][1];

    };

    for (int j=4*(i+1); j<par.nfftpix_ring; j+=j) {

      memcpy(&fftmap_in__pix[par.nfftpix_ring*i+j],
             &fftmap_in__pix[par.nfftpix_ring*i],
             sizeof(fftmap_in__pix)*min(j, par.nfftpix_ring-j));

      memcpy(&fftmap_in__pix[par.nfftpix_ring*(par.nrings-1-i)+j],
             &fftmap_in__pix[par.nfftpix_ring*(par.nrings-1-i)],
             sizeof(fftmap_in__pix)*min(j, par.nfftpix_ring-j));

    };

  };

  for (int i=par.nside-1; i<3*par.nside; i++) {

    fftwf_execute_dft_r2c(forward_plan_map[par.nside-1],
                          &map_in__pix[par.npix_ring*i],
                          &fftmap_in__pix[par.nfftpix_ring*i]);

  };

  rotate_fftrings(fftmap_in__pix, par, -1.0f);

}

void convmap::rotate_fftrings(fftwf_complex* fftmap__pix,
                              const parameter par, const float direction) {

  float                                                     pi_npix;
  float                                                     shiftangle;
  float                                                     sinshift;
  float                                                     cosshift;
  float                                                     temp_real;
  float                                                     temp_imag;

  for (int i=0; i<par.nside-1; i++) {

    pi_npix = direction*PI_f/(float (par.npix_ring));

    for (int j=1, k=1.0f; j<par.nfftpix_ring; j++, k+=1.0f) {

      shiftangle = k*pi_npix;
      sincosf(shiftangle, &sinshift, &cosshift);

      temp_real
        = cosshift * fftmap__pix[par.nfftpix_ring*i+j][0]
        - sinshift * fftmap__pix[par.nfftpix_ring*i+j][1];
      temp_imag
        = sinshift * fftmap__pix[par.nfftpix_ring*i+j][0]
        + cosshift * fftmap__pix[par.nfftpix_ring*i+j][1];
      fftmap__pix[par.nfftpix_ring*i+j][0] = temp_real;
      fftmap__pix[par.nfftpix_ring*i+j][1] = temp_imag;

      temp_real
        = cosshift * fftmap__pix[par.nfftpix_ring*(par.nrings-1-i)+j][0]
        - sinshift * fftmap__pix[par.nfftpix_ring*(par.nrings-1-i)+j][1];
      temp_imag
        = sinshift * fftmap__pix[par.nfftpix_ring*(par.nrings-1-i)+j][0]
        + cosshift * fftmap__pix[par.nfftpix_ring*(par.nrings-1-i)+j][1];
      fftmap__pix[par.nfftpix_ring*(par.nrings-1-i)+j][0] = temp_real;
      fftmap__pix[par.nfftpix_ring*(par.nrings-1-i)+j][1] = temp_imag;

    };

  };

  pi_npix = direction*PI_f/(float (par.npix_ring));

  for (int i=par.nside-1; i<3*par.nside; i+=2) {

    for (int j=1, k=1.0f; j<par.nfftpix_ring; j++, k+=1.0f) {

      shiftangle = k*pi_npix;
      sincosf(shiftangle, &sinshift, &cosshift);

      temp_real
        = cosshift * fftmap__pix[par.nfftpix_ring*i+j][0]
        - sinshift * fftmap__pix[par.nfftpix_ring*i+j][1];
      temp_imag
        = sinshift * fftmap__pix[par.nfftpix_ring*i+j][0]
        + cosshift * fftmap__pix[par.nfftpix_ring*i+j][1];
      fftmap__pix[par.nfftpix_ring*i+j][0] = temp_real;
      fftmap__pix[par.nfftpix_ring*i+j][1] = temp_imag;

    };

  };

}

void convmap::convolve_with_kernel(convkernel kernel, const parameter par) {

  fftwf_complex*                                            fftkernel_ptr;
  fftwf_complex*                                            fftmap_in_ptr;
  fftwf_complex*                                            fftmap_out_ptr;

  for (int i=0; i<par.nrings/2+1; i++) {

    kernel.kernel2grid(i, par);
    kernel.fft_kernel(par);

    for (int j=-kernel.support_rings; j<=kernel.support_rings; j++) {

      if ((0 > i+j) || (par.nrings-1 < i+j)) continue;

      fftkernel_ptr  = &kernel.fftkernel__pix[par.nfftpix_ring*(kernel.support_rings+j)];
      fftmap_in_ptr  = &fftmap_in__pix[par.nfftpix_ring*(i+j)];
      fftmap_out_ptr = &fftmap_out__pix[par.nfftpix_ring*i];

      for (int h=0; h<par.nfftpix_ring; h++) {

        fftmap_out_ptr[h][0] += fftkernel_ptr[h][0] * fftmap_in_ptr[h][0]
          - fftkernel_ptr[h][1] * fftmap_in_ptr[h][1];

        fftmap_out_ptr[h][1] += fftkernel_ptr[h][0] * fftmap_in_ptr[h][1]
          + fftkernel_ptr[h][1] * fftmap_in_ptr[h][0];

      };

    };

    if (par.nrings/2 == i) break;

    for (int j=kernel.support_rings; j>=-kernel.support_rings; j--) {

      if ((0 > par.nrings-1-i+j) || (0 > i-j)) continue;

      fftkernel_ptr  = &kernel.fftkernel__pix[par.nfftpix_ring*(kernel.support_rings-j)];
      fftmap_in_ptr  = &fftmap_in__pix[par.nfftpix_ring*(par.nrings-1-i+j)];
      fftmap_out_ptr = &fftmap_out__pix[par.nfftpix_ring*(par.nrings-1-i)];

      for (int h=0; h<min(par.nfftpix_ring, 2*(i+1)+1); h++) {

        fftmap_out_ptr[h][0] += fftkernel_ptr[h][0] * fftmap_in_ptr[h][0]
          - fftkernel_ptr[h][1] * fftmap_in_ptr[h][1];

        fftmap_out_ptr[h][1] += fftkernel_ptr[h][0] * fftmap_in_ptr[h][1]
          + fftkernel_ptr[h][1] * fftmap_in_ptr[h][0];

      };

    };

  };

  fftkernel_ptr  = NULL;
  fftmap_in_ptr  = NULL;
  fftmap_out_ptr = NULL;

}

void convmap::ifft_map(const convkernel kernel, const parameter par) {

  int                                                       npix;

  rotate_fftrings(fftmap_out__pix, par, 1.0f);

  for (int i=0; i<par.nside-1; i++) {

    npix = 4*(i+1);

    for (int j=npix; j<par.nfftpix_ring; j++) {

      fftmap_out__pix[par.nfftpix_ring*i+j%npix][0]
        += fftmap_out__pix[par.nfftpix_ring*i+j][0];
      fftmap_out__pix[par.nfftpix_ring*i+j%npix][1]
        += fftmap_out__pix[par.nfftpix_ring*i+j][1];

      fftmap_out__pix[par.nfftpix_ring*(par.nrings-1-i)+j%npix][0]
        += fftmap_out__pix[par.nfftpix_ring*(par.nrings-1-i)+j][0];
      fftmap_out__pix[par.nfftpix_ring*(par.nrings-1-i)+j%npix][1]
        += fftmap_out__pix[par.nfftpix_ring*(par.nrings-1-i)+j][1];

    };

    npix = 2*(i+1);
    fftmap_out__pix[par.nfftpix_ring*i+npix][1]              = 0.0f;
    fftmap_out__pix[par.nfftpix_ring*(par.nrings-i)+npix][1] = 0.0f;

    fftwf_execute_dft_c2r(backward_plan_map[i],
                          &fftmap_out__pix[par.nfftpix_ring*i],
                          &map_out__pix[par.npix_ring*i]);

    fftwf_execute_dft_c2r(backward_plan_map[i],
                          &fftmap_out__pix[par.nfftpix_ring*(par.nrings-1-i)],
                          &map_out__pix[par.npix_ring*(par.nrings-1-i)]);

  };

  for (int i=par.nside-1; i<3*par.nside; i++) {

    fftmap_out__pix[par.nfftpix_ring*(i+1)-1][1]          = 0.0f;
    fftmap_out__pix[par.nfftpix_ring*(par.nrings-i)-1][1] = 0.0f;

    fftwf_execute_dft_c2r(backward_plan_map[par.nside-1],
                          &fftmap_out__pix[par.nfftpix_ring*i],
                          &map_out__pix[par.npix_ring*i]);

  };

}

void convmap::grid2map(const convkernel kernel, const parameter par) {

  int                                                       npix_map_ring;
  int                                                       pix_offset;
  float                                                     normalization;

  pix_offset    = 0;
  npix_map_ring = 0;
  normalization = 4.0f*PI_f/(float(par.npix) * float (par.npix_ring));

  for (int i=0; i<par.nside-1; i++) {

    npix_map_ring = 4*(i+1);

    for (int j=0; j<npix_map_ring; j++) {
      map_I[pix_offset+j] = normalization * map_out__pix[par.npix_ring*i+j];
    };

    pix_offset += npix_map_ring;

  };

  for (int i=par.nside-1; i<3*par.nside; i++) {
    for (int j=0; j<par.npix_ring; j++) {
      map_I[pix_offset] = normalization * map_out__pix[par.npix_ring*i+j];
      pix_offset++;
    };
  };

  for (int i=3*par.nside; i<4*par.nside-1; i++) {

    npix_map_ring = 4*(4*par.nside-1 - i);

    for (int j=0; j<npix_map_ring; j++) {
      map_I[pix_offset+j] = normalization * map_out__pix[par.npix_ring*i+j];
    };

    pix_offset += npix_map_ring;

  };

}

void convmap::get_first_ring(const skymap map, convkernel kernel,
                             const parameter par) {

  int                                                       npix_in_ring;
  int                                                       offset_ring;
  float                                                     kernel_pix_0;
  float                                                     kernel_pix_1;
  float                                                     kernel_pix_2;
  float                                                     kernel_pix_3;
  float                                                     normalization;
  double                                                    angle_sqr_0;
  double                                                    angle_sqr_1;
  double                                                    angle_sqr_2;
  double                                                    angle_sqr_3;
  double                                                    sint1_sqr;
  double                                                    sint2_sqr;
  double                                                    anglesum;
  double                                                    angledif;
  double                                                    phi_pix;
  double                                                    phi_0;
  double                                                    phi_1;
  double                                                    phi_2;
  double                                                    phi_3;

  offset_ring       = 0;
  map_I[0]          = 0.0;
  map_I[1]          = 0.0;
  map_I[2]          = 0.0;
  map_I[3]          = 0.0;
  map_I[par.npix-4] = 0.0;
  map_I[par.npix-3] = 0.0;
  map_I[par.npix-2] = 0.0;
  map_I[par.npix-1] = 0.0;

  phi_0 = 0.25 * PI_d;
  phi_1 = 3.0 * phi_0;
  phi_2 = 5.0 * phi_0;
  phi_3 = 7.0 * phi_0;

  for (int i=0; i<kernel.support_rings+1; i++) {

    sint1_sqr = sin(0.5*(kernel.theta__ring[i] + kernel.theta__ring[0]));
    sint2_sqr = sin(0.5*(kernel.theta__ring[i] - kernel.theta__ring[0]));
    sint1_sqr = sint1_sqr * sint1_sqr;
    sint2_sqr = sint2_sqr * sint2_sqr;
    anglesum  = 2.0*(sint1_sqr + sint2_sqr);
    angledif  = 2.0*(sint1_sqr - sint2_sqr);

    if (i < par.nside-1) {
      npix_in_ring = 4*(i+1);
    } else if (i < 3*par.nside) {
      npix_in_ring = par.npix_ring;
    } else {
      npix_in_ring = 4*(par.nrings-i);
    };

    phi_pix = PI_d/(double (npix_in_ring));

    for (int j=0; j<npix_in_ring; j++) {

      angle_sqr_0  = anglesum - angledif*cos((double (2*j+1)) * phi_pix - phi_0);
      angle_sqr_1  = anglesum - angledif*cos((double (2*j+1)) * phi_pix - phi_1);
      angle_sqr_2  = anglesum - angledif*cos((double (2*j+1)) * phi_pix - phi_2);
      angle_sqr_3  = anglesum - angledif*cos((double (2*j+1)) * phi_pix - phi_3);
      kernel_pix_0 = kernel.interp_kernel(angle_sqr_0);
      kernel_pix_1 = kernel.interp_kernel(angle_sqr_1);
      kernel_pix_2 = kernel.interp_kernel(angle_sqr_2);
      kernel_pix_3 = kernel.interp_kernel(angle_sqr_3);
      map_I[0]    += kernel_pix_0 * map.map_I[offset_ring+j];
      map_I[1]    += kernel_pix_1 * map.map_I[offset_ring+j];
      map_I[2]    += kernel_pix_2 * map.map_I[offset_ring+j];
      map_I[3]    += kernel_pix_3 * map.map_I[offset_ring+j];
      map_I[par.npix-4] += kernel_pix_0
        * map.map_I[par.npix-offset_ring-npix_in_ring+j];
      map_I[par.npix-3] += kernel_pix_1
        * map.map_I[par.npix-offset_ring-npix_in_ring+j];
      map_I[par.npix-2] += kernel_pix_2
        * map.map_I[par.npix-offset_ring-npix_in_ring+j];
      map_I[par.npix-1] += kernel_pix_3
        * map.map_I[par.npix-offset_ring-npix_in_ring+j];

    };

    offset_ring += npix_in_ring;

  };

  normalization = 4.0f*PI_f/(float (par.npix));

  map_I[0]          *= normalization;
  map_I[1]          *= normalization;
  map_I[2]          *= normalization;
  map_I[3]          *= normalization;
  map_I[par.npix-4] *= normalization;
  map_I[par.npix-3] *= normalization;
  map_I[par.npix-2] *= normalization;
  map_I[par.npix-1] *= normalization;

}

void convmap::convolve(const skymap map, convkernel kernel,
                       const parameter par) {

  uint64                                                  wct_start;
  uint64                                                  wct_stop;

  wct_start = rdtsc();

  if (par.do_gpu) {

    allocate_fft_map_gpu(kernel, par);

    map2grid(map, kernel, par);

    convolve_gpu(map_out__pix,
                 map_in__pix,
                 kernel.theta__ring,
                 kernel.support_rad,
                 kernel.conversion_factor,
                 kernel.delta_angle,
                 kernel.kernel__alpha,
                 kernel.support_rings,
                 kernel.nr_kernel_nodes,
                 par.nrings,
                 par.nside,
                 par.npix_ring,
                 par.nfftpix_ring);

    grid2map(kernel, par);

    get_first_ring(map, kernel, par);

  } else {

    allocate_fft_map(kernel, par);

    map2grid(map, kernel, par);

    fft_map(kernel, par);

    convolve_with_kernel(kernel, par);

    ifft_map(kernel, par);

    grid2map(kernel, par);

    get_first_ring(map, kernel, par);

  };

  wct_stop = rdtsc();

  cout <<  (wct_stop - wct_start) / 2.4E9 << endl;

}

//-----------------------------------------------------------------------
