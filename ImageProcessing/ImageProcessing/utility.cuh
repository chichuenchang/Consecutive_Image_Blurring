#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include "math_functions.h"

extern "C"
{

	//utility, call kernel
	__device__ float gausFunc(float x);
	__device__ float recFunc(float x);
	__device__ float trigFunc(float x);



};