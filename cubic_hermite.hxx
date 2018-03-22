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

int chfev(double, double, double, double, double, double, int, double*,
          double*, int*);

void spline_pchip_set(int, double*, double*, double*);

void spline_pchip_val(int, double*, double*, double*, int,
                      double*, double*);

double pchst(double&, double&);
int i4_max(int, int);
int i4_min(int, int);
double r8_max(double, double);
double r8_min(double, double);
