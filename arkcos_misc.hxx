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

extern "C" uint64 rdtsc();

float ring2theta(const int&, const int&, const int&);
double ring2theta_double(const int&, const int&, const int&);

double randn_trig(double, double);
float randn_trig(float, float);

std::string int2string(const int);

float* alloc_pseudo2d_arr_float(const int&, const int&);
void free_pseudo2d_arr_float(float*);

fftwf_complex* alloc_pseudo2d_arr_fftwf(const int&, const int&);
void free_pseudo2d_arr_fftwf(fftwf_complex*);
