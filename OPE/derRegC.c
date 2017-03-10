/*
* @Author: Vu Van Tu
* @Date:   2017-01-02 13:10:18
* @Last Modified by:   Vu Van Tu
* @Last Modified time: 2017-01-02 17:54:24
*/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mex.h>
#include <math.h>
#include "matrix.h"
#include "funElementReg.c"

/*
-------------------------- Derivation of Regulazation -----------------------------

 f = \sum_i r_i(x)

 Usage (in matlab):
 [f]=derRegC(x, n, lambda, theta, type, eps); n is the dimension of the vector x

 type = 1: Capped L1 regularizer (CapL1) 
           r_i(x) = lambda*\min(|x_i|,theta), (theta > 0, lambda >= 0)

 type = 2: Log Sum Penalty (LSP)
           r_i(x) = lambda*\sum_i log(1 + |x_i|/theta), (theta > 0, lambda >= 0)

 type = 3: Smoothly Clipped Absolute Deviation (SCAD)
           r_i(x) = lambda*|x_i|, if |x_i|<=lambda
           r_i(x) = (-x_i^2 + 2*theta*lambda*|x_i| - lambda^2)/(2(theta - 1)), if lambda<=|x_i|<=theta*lambda
           r_i(x) = 0.5*(theta + 1)*lambda^2, if |x_i| > theta*lambda, (theta > 2, lambda >= 0)
				  

 type = 4: Minimax Concave Penalty (MCP)
           r_i(x) = lambda*|x_i| - 0.5*x_i^2/theta, if |x_i|<=theta*lambda
           r_i(x) = 0.5*theta*lambda^2, if |x_i| > theta*lambda, (theta > 0, lambda >= 0)

 default: type = 1

*/
double max(double a, double b){
	return a > b ? a : b;
}

void derCapL1(double *f, double *x, long n, double lambda, double theta, double eps)
{
	long i;
  double rand_value = 0;
	for(i=0;i<n;i++) {
		if(x[i] == theta){
			f[i] = max(lambda, 0);
		} else if(x[i] > 0 && x[i] < theta){
			f[i] = lambda;
		} else if(x[i] == 0){
			f[i] = max(lambda, -lambda);
		} else if(x[i] < 0 && x[i] > -theta){
			f[i] = -lambda;
		} else if(x[i] == -theta){
			f[i] = max(-lambda,0);
		} else {
			f[i] = eps;
		}
	}
	return;
}
          
void derLSP(double *f, double *x, long n, double lambda, double theta, double eps)
{
    long i;
    double delta = 0;
	for(i=0;i<n;i++) { 
		if(x[i] > 0){
			f[i] = lambda * (x[i] + theta);
		} else if(x[i] == 0) {
			f[i] = max(lambda*(x[i]+theta), -lambda*(x[i]+theta));
		} else {
			f[i] = -lambda * (x[i] + theta);
		}
	}
	return;
}

void derSCAD(double *f, double *x, long n, double lambda, double theta, double eps)
{
  long i;
	double u = theta*lambda;
	double delta;

	for(i=0;i<n;i++) {
		if(x[i] == u){
			f[i] = max(0, (-x[i]+u)/(theta-1));
		} else if(x[i] < u && x[i] > lambda){
			f[i] = (-x[i]+u)/(theta-1);
		} else if(x[i] == lambda){
			f[i] = max((-x[i]+u)/(theta-1), lambda);
		} else if(x[i] < lambda && x[i] > 0){
			f[i] = lambda;
		} else if(x[i] == 0){
			f[i] = max(lambda, -lambda);
		} else if(x[i] < 0 && x[i] > -lambda){
			f[i] = -lambda;
		} else if(x[i] == -lambda){
			f[i] = max(-lambda, (-x[i]-u)/(theta-1));
		} else if(x[i] < -lambda && x[i] > -u) {
			f[i] = (-x[i]-u)/(theta-1); 
		} else if(x[i] == -u){
			f[i] = max((-x[i]-u)/(theta-1), 0);
		} else{
			f[i] = 0;
		}
	}
	return;
}

void derMCP(double *f, double *x, long n, double lambda, double theta, double eps)
{
	long i;
	double u = theta*lambda;
  double delta = 0;

  for(i=0;i<n;i++) { 
  	if(x[i] == u){
  		f[i] = max(0, lambda - x[i]/theta);
  	} else if(x[i] < u && x[i] > 0){
  		f[i] = lambda - x[i]/theta;
  	} else if(x[i] == 0){
  		f[i] = max(lambda - x[i]/theta, -lambda - x[i]/theta);
  	} else if(x[i] < 0 && x[i] > -u){
  		f[i] = -lambda - x[i]/theta;
  	} else if(x[i] == -u){
  		f[i] = max(-lambda - x[i]/theta, 0);
  	} else{
  		f[i] = 0;
  	}
	}
	return;
}

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
  /*set up input arguments */
  double* x       =            mxGetPr(prhs[0]);
  long     n      =      (long)mxGetScalar(prhs[1]);
  double  lambda  =            mxGetScalar(prhs[2]);
	double  theta   =            mxGetScalar(prhs[3]);
	double  eps 	= 			 mxGetScalar(prhs[4]);
	int     type    =       (int)mxGetScalar(prhs[5]);
    
  double *f;


  /* set up output arguments */
  plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);
  
  f=mxGetPr(plhs[0]);
	
	switch (type) {
	case 1:
		derCapL1(f, x, n, lambda, theta, eps);
		break;
	case 2:
		derLSP(f, x, n, lambda, theta, eps);
		break;
	case 3:
		derSCAD(f, x, n, lambda, theta, eps);
		break;
	case 4:
		derMCP(f, x, n, lambda, theta, eps);
		break;
	default:
		derCapL1(f, x, n, lambda, theta, eps);
	}
}


