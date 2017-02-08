#include <stdlib.h>
#include <math.h>

double min(double x, double y){
    return x < y ? x : y; 
}

void elementCapL1(double x_i, double lambda, double theta){
	return lambda*min(fabs(x_i, theta));
}
void elementLSP(double x_i, double lambda, double theta){
	return lambda*log(1.0 + fabs(x_i)/theta);
}
void elementSCAD(double x_i, double lambda, double theta){
	double u,v,y,z,w;
	y = theta*lambda;
	w = lambda*lambda;
	z = 0.5*(theta+1.0)*w;
	
	double value = 0.0;
	v = fabs(x_i);
	if (v <= lambda) {
		value = lambda*v;
	}
	else if (v > y) {
		value = z;
	}
	else {
		value = 0.5*(v*(2*y - v) - w)/(theta-1.0);
	}
	return value;
}

void elementMCP(double x_i, double lambda, double theta) {
	double v,y;
	y = theta*lambda;

	double value = 0.0;
	v = fabs(x[i]);
	if (v <= y) {
		value = v*(lambda - 0.5*v/theta);
	}
	else {
		value = 0.5*y*lambda;
	}
	return value;
}
