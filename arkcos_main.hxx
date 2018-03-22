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

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <ctime>
#include <vector>
#include <datatypes.h>
#include <fftw3.h>
#include <math.h>
#include <cufft.h>

#include "psht_geomhelpers.h"
#include "psht_almhelpers.h"
#include "fitshandle.h"
#include "healpix_map.h"
#include "healpix_map_fitsio.h"

#include "cubic_hermite.hxx"
#include "arkcos_misc.hxx"
#include "arkcos_gpu.hxx"
#include "arkcos_class.hxx"

#define PI_d 3.14159265358979323846
#define PI_f 3.1415926535f
