#include "utility.cuh"

__device__ float gausFunc(float x) {

	float sig = 2.0f, pi = 3.1415926f, niu = 0.0f;
	//gaussian equation
	return 1 / (sig * powf(2 * pi, 0.5)) * expf(-0.5 * ((x - niu) / sig) * ((x - niu) / sig));
}

__device__ float recFunc(float x) {

	float d = 0.5;
	if (x < d) return 1.0;
	else if (x = d) return 0.5;
	else return 0;
}

__device__ float trigFunc(float x) {

	const float k = 0.3;
	float y = 1 - k * fabs(x);

	if (y < 0) y = 0;
	return y;
}