#include "funElementReg.c"

void funCapL1(double *f, double *x, long n, double lambda, double theta)
{
    long i;
	double u = 0.0;;
	for(i=0;i<n;i++){   
		u += elementCapL1(x[i], lambda, theta);
	}
	*f = u;
	return;
}

void funLSP(double *f, double *x, long n, double lambda, double theta)
{
    long i;
	double u = 0.0;
	for(i=0;i<n;i++){ 
		u += elementLSP(x[i], lambda, theta);
	}
	*f = u;
	return;
}


void funSCAD(double *f, double *x, long n, double lambda, double theta)
{
    long i;
	u = 0.0;
	for(i=0;i<n;i++) { 
		u += elementSCAD(x[i], lambda, theta);
	}
	*f = u;
	return;
}

void funMCP(double *f, double *x, long n, double lambda, double theta)
{
    long i;
	u = 0.0;
    for(i=0;i<n;i++){ 
		u += elementMCP(x[i], lambda, theta);
	}
	*f = u;
	return;
}
