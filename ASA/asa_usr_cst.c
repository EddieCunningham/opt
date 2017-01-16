/***********************************************************************
* Adaptive Simulated Annealing (ASA)
* Lester Ingber <ingber@ingber.com>
* Copyright (c) 1987-2016 Lester Ingber.  All Rights Reserved.
* ASA-LICENSE file has the license that must be included with ASA code.
***********************************************************************/

 /* $Id: asa_usr_cst.c,v 30.21 2016/02/02 15:49:43 ingber Exp ingber $ */

 /* asa_usr_cst.c for Adaptive Simulated Annealing */

#include "asa_usr.h"

#if COST_FILE

 /* Note that this is a trimmed version of the ASA_TEST problem.
    A version of this cost_function with more documentation and hooks for
    various templates is in asa_usr.c. */

 /* If you use this file to define your cost_function (the default),
    insert the body of your cost function just above the line
    "#if ASA_TEST" below.  (The default of ASA_TEST is FALSE.)

    If you read in information via the asa_opt file (the default),
    define *parameter_dimension and
    parameter_lower_bound[.], parameter_upper_bound[.], parameter_int_real[.]
    for each parameter at the bottom of asa_opt.

    The minimum you need to do here is to use
    x[0], ..., x[*parameter_dimension-1]
    for your parameters and to return the value of your cost function.  */

#if HAVE_ANSI
double
cost_function (double *x,
               double *parameter_lower_bound,
               double *parameter_upper_bound,
               double *cost_tangents,
               double *cost_curvature,
               ALLOC_INT * parameter_dimension,
               int *parameter_int_real,
               int *cost_flag, int *exit_code, USER_DEFINES * USER_OPTIONS)
#else
double
cost_function (x,
               parameter_lower_bound,
               parameter_upper_bound,
               cost_tangents,
               cost_curvature,
               parameter_dimension,
               parameter_int_real, cost_flag, exit_code, USER_OPTIONS)
     double *x;
     double *parameter_lower_bound;
     double *parameter_upper_bound;
     double *cost_tangents;
     double *cost_curvature;
     ALLOC_INT *parameter_dimension;
     int *parameter_int_real;
     int *cost_flag;
     int *exit_code;
     USER_DEFINES *USER_OPTIONS;
#endif
{

	struct f {
		double *_x;

		f(double *x) {
			_x = x;
		}

		double eval(int __x) {
			double ans = 0.5;
			int N = 2;

			for(int i=0; i<N; ++i) {
				double a = this->_x[8*i+0];
				double b = this->_x[8*i+1];
				double c = this->_x[8*i+2];
				double d = this->_x[8*i+3];
				double e = this->_x[8*i+4];
				double f = this->_x[8*i+5];
				double g = this->_x[8*i+6];
				double h = this->_x[8*i+7];

				ans += int(cos(a*int(__x))) + int(sin(b*int(__x))) + int(sin(c*int(__x))*sin(d*int(__x))) + int(cos(e*int(__x))*cos(f*int(__x))) + int(sin(g*int(__x))*sin(h*int(__x)));
			}
			ans /= (2.0*N);
			return ans;
		}
	};

	f _f(x);

	double ans = 0;


	ans += pow(_f.eval(0) - 1.00,2);
	// ans += pow(_f.eval(1) - 0.50,2);
	// ans += pow(_f.eval(2) - 0.00,2);
	// ans += pow(_f.eval(3) - 0.50,2);
	// ans += pow(_f.eval(4) - 0.25,2);
	// ans += pow(_f.eval(5) - 0.00,2);
	// ans += pow(_f.eval(6) - 0.00,2);
	// ans += pow(_f.eval(7) - 0.00,2);
	// ans += pow(_f.eval(8) - 0.00,2);
	// ans += pow(_f.eval(9) - 0.00,2);
	// ans += pow(_f.eval(10) - 0.50,2);
	// ans += pow(_f.eval(11) - 1.00,2);
	// ans += pow(_f.eval(12) - 0.50,2);
	// ans += pow(_f.eval(13) - 0.50,2);
	// ans += pow(_f.eval(14) - 0.50,2);
	// ans += pow(_f.eval(15) - 1.00,2);
	// ans += pow(_f.eval(16) - 0.50,2);
	// ans += pow(_f.eval(17) - 0.00,2);
	// ans += pow(_f.eval(18) - 0.00,2);
	// ans += pow(_f.eval(19) - 0.00,2);
	// ans += pow(_f.eval(20) - 0.00,2);
	// ans += pow(_f.eval(21) - 0.00,2);
	// ans += pow(_f.eval(22) - 0.25,2);
	// ans += pow(_f.eval(23) - 0.50,2);
	// ans += pow(_f.eval(24) - 0.00,2);
	// ans += pow(_f.eval(25) - 0.50,2);
	// ans += pow(_f.eval(26) - 1.00,2);




	return ans;


  /* *** Insert the body of your cost function here, or warnings
   * may occur if COST_FILE = TRUE & ASA_TEST != TRUE ***
   * Include ADAPTIVE_OPTIONS below if required */
#if ASA_TEST
#else
#if ADAPTIVE_OPTIONS
  adaptive_options (USER_OPTIONS);
#endif
#endif

#if ASA_TEST
  double q_n, d_i, s_i, t_i, z_i, c_r;
  int k_i;
  ALLOC_INT i, j;
  static LONG_INT funevals = 0;

#if ADAPTIVE_OPTIONS
  adaptive_options (USER_OPTIONS);
#endif

  s_i = 0.2;
  t_i = 0.05;
  c_r = 0.15;

  q_n = 0.0;
  for (i = 0; i < *parameter_dimension; ++i) {
    j = i % 4;
    switch (j) {
    case 0:
      d_i = 1.0;
      break;
    case 1:
      d_i = 1000.0;
      break;
    case 2:
      d_i = 10.0;
      break;
    default:
      d_i = 100.0;
    }
    if (x[i] > 0.0) {
      k_i = (int) (x[i] / s_i + 0.5);
    } else if (x[i] < 0.0) {
      k_i = (int) (x[i] / s_i - 0.5);
    } else {
      k_i = 0;
    }

    if (fabs (k_i * s_i - x[i]) < t_i) {
      if (k_i < 0) {
        z_i = k_i * s_i + t_i;
      } else if (k_i > 0) {
        z_i = k_i * s_i - t_i;
      } else {
        z_i = 0.0;
      }
      q_n += c_r * d_i * z_i * z_i;
    } else {
      q_n += d_i * x[i] * x[i];
    }
  }
  funevals = funevals + 1;

  *cost_flag = TRUE;

#if FALSE                       /* may set to TRUE if printf() is active */
#if TIME_CALC
  if ((PRINT_FREQUENCY > 0) && ((funevals % PRINT_FREQUENCY) == 0)) {
    printf ("funevals = %ld  ", funevals);
    print_time ("", stdout);
  }
#endif
#endif

#if ASA_FUZZY
  if (*cost_flag == TRUE
      && (USER_OPTIONS->Locate_Cost == 2 || USER_OPTIONS->Locate_Cost == 3
          || USER_OPTIONS->Locate_Cost == 4)) {
    FuzzyControl (USER_OPTIONS, x, q_n, *parameter_dimension);
  }
#endif /* ASA_FUZZY */

  return (q_n);
#endif /* ASA_TEST */
}
#endif /* COST_FILE */
