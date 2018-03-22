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
// Timing
//-----------------------------------------------------------------------

extern "C" {

  uint64 rdtsc() {
    uint32 lo, hi;
    __asm__ __volatile__ (      // serialize
                          "xorl %%eax,%%eax \n        cpuid"
                          ::: "%rax", "%rbx", "%rcx", "%rdx");
/* We cannot use "=A", since this would use %rax on x86_64 */
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return (uint64)hi << 32 | lo;
  }

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Compute theta for a given ring: float
//-----------------------------------------------------------------------

float ring2theta(const int& in_ring, const int& nside,
                 const int& nrings) {

  int                                                       ring;
  int                                                       mirror_ring;

  ring = in_ring + 1;

  if (ring <= nside-1) {

    return acos(1.0f - float(ring*ring)/float(3*nside*nside));

  } else if (ring <= 2*nside) {

    return acos(4.0f/3.0f - 2.0f*float(ring)/float(3*nside));

  };

  mirror_ring = nrings - in_ring;

  if (mirror_ring <= nside-1) {

    return PI_f - acos(1.0f - float(mirror_ring*mirror_ring)/float(3*nside*nside));

  } else if (mirror_ring < 2*nside) {

    return PI_f - acos(4.0f/3.0f - 2.0f*float(mirror_ring)/float(3*nside));

  } else {
    cout << "Error: ring -> theta" << endl;
    abort();
  };

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Compute theta for a given ring: double
//-----------------------------------------------------------------------

double ring2theta_double(const int& in_ring, const int& nside,
                         const int& nrings) {

  int                                                       ring;
  int                                                       mirror_ring;

  ring = in_ring + 1;

  if (ring <= nside-1) {

    return acos(1.0 - (double (ring*ring))/(double (3*nside*nside)));

  } else if (ring <= 2*nside) {

    return acos(4.0/3.0 - 2.0*(double (ring))/(double (3*nside)));

  };

  mirror_ring = nrings - in_ring;

  if (mirror_ring <= nside-1) {

    return PI_d - acos(1.0 - (double (mirror_ring*mirror_ring))/(double (3*nside*nside)));

  } else if (mirror_ring < 2*nside) {

    return PI_d - acos(4.0/3.0 - 2.0*(double (mirror_ring))/(double (3*nside)));

  } else {
    cout << "Error: ring -> theta" << endl;
    abort();
  };

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Generate filename with integer count
//-----------------------------------------------------------------------

string int2string(const int number) {

  std::stringstream stream;

  stream << number;

  return stream.str();

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
//	Box-Mueller, double precision
//-----------------------------------------------------------------------

double randn_trig(double mu=0.0, double sigma=1.0) {
  static bool deviateAvailable=false;	//	flag
  static float storedDeviate;			//	deviate from previous calculation
  double dist, angle;

//	If no deviate has been stored, the standard Box-Muller transformation is
//	performed, producing two independent normally-distributed random
//	deviates.  One is stored for the next round, and one is returned.
  if (!deviateAvailable) {

//	choose a pair of uniformly distributed deviates, one for the
//	distance and one for the angle, and perform transformations
    dist=sqrt( -2.0 * log(double(rand()) / double(RAND_MAX)) );
    angle=2.0 * PI_d * (double(rand()) / double(RAND_MAX));

//	calculate and store first deviate and set flag
    storedDeviate=dist*cos(angle);
    deviateAvailable=true;

//	calcaulate return second deviate
    return dist * sin(angle) * sigma + mu;
  }

//	If a deviate is available from a previous call to this function, it is
//	returned, and the flag is set to false.
  else {
    deviateAvailable=false;
    return storedDeviate*sigma + mu;
  }

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
//	Box-Mueller, single precision
//-----------------------------------------------------------------------

float randn_trig(float mu=0.0, float sigma=1.0) {
  static bool deviateAvailable=false;	//	flag
  static float storedDeviate;			//	deviate from previous calculation
  float dist, angle;

//	If no deviate has been stored, the standard Box-Muller transformation is
//	performed, producing two independent normally-distributed random
//	deviates.  One is stored for the next round, and one is returned.
  if (!deviateAvailable) {

//	choose a pair of uniformly distributed deviates, one for the
//	distance and one for the angle, and perform transformations
    dist  = sqrt( -2.0f * logf(float(rand()) / float(RAND_MAX)) );
    angle = 2.0f * PI_f * (float(rand()) / float(RAND_MAX));

//	calculate and store first deviate and set flag
    storedDeviate    = dist * cosf(angle);
    deviateAvailable = true;

//	calcaulate return second deviate
    return dist * sinf(angle) * sigma + mu;
  }

//	If a deviate is available from a previous call to this function, it is
//	returned, and the flag is set to false.
  else {
    deviateAvailable = false;
    return storedDeviate*sigma + mu;
  }

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Pseudo-2D array allocation: float
//-----------------------------------------------------------------------

float* alloc_pseudo2d_arr_float(const int& nrows, const int& ncols) {

//-----------------------------------------------------------------------

  float*                                                    array;

  array = (float*) fftwf_malloc(sizeof(float)*nrows*ncols);

  memset(array, 0, sizeof(float)*nrows*ncols);

  return array;

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Pseudo-2D array deallocation: float
//-----------------------------------------------------------------------

void free_pseudo2d_arr_float(float* array) {

//-----------------------------------------------------------------------

  fftwf_free(array);

  array = NULL;

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Pseudo-2D array allocation: fftwf_complex
//-----------------------------------------------------------------------

fftwf_complex* alloc_pseudo2d_arr_fftwf(const int& nrows,
                                        const int& ncols) {

//-----------------------------------------------------------------------

  fftwf_complex*                                            array;

  array = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex)*nrows*ncols);

  memset(array, 0, sizeof(fftwf_complex)*nrows*ncols);

  return array;

}

//-----------------------------------------------------------------------



//-----------------------------------------------------------------------
// Pseudo-2D array deallocation: fftwf_complex
//-----------------------------------------------------------------------

void free_pseudo2d_arr_fftwf(fftwf_complex* array) {

//-----------------------------------------------------------------------

  fftwf_free(array);

  array = NULL;

}

//-----------------------------------------------------------------------
