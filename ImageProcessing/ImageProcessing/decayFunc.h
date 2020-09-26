#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include "math_functions.h"
//#include "utility.cuh"

class decayFunc
{
private:
	unsigned char* d_imgIn;
	unsigned char* d_imgOut;
	int imgW, imgH, numChannel;
	int f_col, f_row;
	int f_ind;

public:
	
	//constructor
	//__host__ decayFunc(const char* imgAdr);

	//constructor
	//__host__ decayFunc(unsigned char* passImgIn, unsigned char* passImgOut, int passW, int passH, int passNumChannel);
	
	//kernel
	
	//utility, call kernel
	__device__ static float gausFunc(float x);
	__device__ static float recFunc(float x);
	__device__ static float trigFunc(float x);



};

