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

int main() {

  string                                                    file_in_map;
  string                                                    file_in_kernel;
  string                                                    file_out_map;
  parameter                                                 par;
  skymap                                                    map_in;
  convmap                                                   map_out;
  convkernel                                                kernel;

  file_in_map    = "/path/to/map.fits";
  file_in_kernel = "/path/to/kernel.dat";
  file_out_map   = "!convolved_map.fits";

//       nside  lmax  mmax
  par.init(512, 1024, 1024, false);

// Compute convolution on GPU:
  par.do_gpu = true;

  map_in.init(par);
  map_out.init(par);
  kernel.init(file_in_kernel, par);

  map_in.fits2map(file_in_map, par);

  map_out.convolve(map_in, kernel, par);

  map_out.map2fits(file_out_map, par);

  par.free();
  map_in.free();
  map_out.free();
  kernel.free();

  return 0;

}
