#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mex.h>
#include <math.h>
#include "matrix.h"

/*
-------------------------- Function derRegC -----------------------------
Calcuate the derivation of function
	f = \sum_i r_i(x)

 Usage (in matlab):
 [f]=derRegC(x, n, lambda, theta,type); n is the dimension of the vector d

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

-------------------------- Function funRegC -----------------------------

-------------------------- Reference -----------------------------------------

[1] Pinghua Gong, Changshui Zhang, Zhaosong Lu, Jianhua Huang, Jieping Ye,
    A General Iterative Shrinkage and Thresholding Algorithm for Non-convex
    Regularized Optimization Problems. ICML 2013.

-----------------------------------------------------------------------------

   Copyright (C) 2012-2013 Pinghua Gong

   For any problem, please contact Pinghua Gong via pinghuag@gmail.com
*/
int max(double x, double y){
    if(x>=y) return 1;
    return 0;
}
int min(double x, double y){
    if(x<y) return 1;
    return 0;
}

void funCapL1(double *f, double x,double lambda, double theta)
{
    if(x == 0 || fabs(x) == theta){
		*f = 0;
		return;
	}
	if(x > theta || x < - theta){
		*f = 0;
		return;
	}
	if(x > -theta && x < 0){
		*f = -1;
		return;
	}
	if(x > 0 && x < theta){
		*f = 1;
		return;
	}
}

void funLSP(double *f, double x, double lambda, double theta)
{
	if(x > 0){
		*f = lambda * x / (x + theta);
	} 
	else if(x < 0){
		*f = lambda * x / (x - theta);
	}
	else {
		*f = 0;
	}
	return;
}

void funSCAD(double *f, double x, double lambda, double theta)
{
	double tmp = theta*lambda;

	if(fabs(x) >= tmp){
		*f = 0;
		return;
	}
	else if(x > lambda && x < tmp){
		*f = (-0.5*x + 2*tmp) / (2*theta-2);
		return;
	}
	else if(x > 0 && x < lambda){
		*f = lambda;
		return;
	}
	else if(x < 0 && x > -lambda){
		*f = -lambda;
	}
	else if(x < -lambda && x > -tmp){
		*f = (-0.5*x - 2*tmp) / (2*theta-2);
		return;
	}
	else{
		*f = 0;
		return;
	}
}

void funMCP(double *f, double x, double lambda, double theta)
{
    if(fabs(x) > lambda){
    	*f = 0;
    }
    else if(x < 0 && x >= - theta * lambda){
    	*f = -lambda - x / theta; 
    }
    else if(x > 0 && x <= theta *lambda){
    	*f = lambda - x/theta;
    }
    else if(x ==0){
    	*f = 0;
    }
    return;
}




void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /*set up input arguments */
    double x       =       (long)mxGetScalar(prhs[0]);
    double  lambda  =            mxGetScalar(prhs[1]);
	double  theta   =            mxGetScalar(prhs[2]);
	int     type    =       (int)mxGetScalar(prhs[3]);
    
    double *f;


    /* set up output arguments */
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    
    f=mxGetPr(plhs[0]);

	switch (type)
	{
	case 1:
		funCapL1(f, x,lambda, theta);
		break;
	case 2:
		funLSP(f, x, lambda, theta);
		break;
	case 3:
		funSCAD(f, x, lambda, theta);
		break;
	case 4:
		funMCP(f, x, lambda, theta);
		break;
	default:
		funCapL1(f, x, lambda, theta);
	}
}


