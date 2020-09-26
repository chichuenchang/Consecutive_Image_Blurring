#include "decayFunc.h"



__device__ float decayFunc::gausFunc(float x) {

	float sig = 2.0f, pi = 3.1415926f, niu = 0.0f;
	//gaussian equation
	return 1 / (sig * powf(2 * pi, 0.5)) * expf(-0.5 * ((x - niu) / sig) * ((x - niu) / sig));
}

__device__ float decayFunc::recFunc(float x) {

	float d = 0.5;
	if (x < d) return 1.0;
	else if (x = d) return 0.5;
	else return 0;
}

__device__ float decayFunc::trigFunc(float x) {

	const float k = 0.3;
	float y = 1 - k * fabs(x);

	if (y < 0) y = 0;
	return y;
}

__global__ void imgKernel(unsigned char* d_imgIn, unsigned char* d_imgOut, int w, int h, int numChan) {

	int f_col = blockIdx.x * blockDim.x + threadIdx.x;
	//int f_row = blockIdx.y * blockDim.y + threadIdx.y;
	//int offset = numChan * (f_row * w + f_col);
	//float gray = (0.299f - 0.01f) * (float)d_imgIn[offset] + (0.587f - 0.01f) * (float)d_imgIn[offset + 1] + (0.144f - 0.01f) * (float)d_imgIn[offset + 2];
	//d_imgOut[offset] = (unsigned char)gray;
	//d_imgOut[offset + 1] = (unsigned char)gray;
	//d_imgOut[offset + 2] = (unsigned char)gray;

}


__global__ void ImgProcKernel_Abandon(unsigned char* d_imgIn, unsigned char* d_imgOut, int width, int height, int components)
{
	int f_col = blockIdx.x * blockDim.x + threadIdx.x;
	int f_row = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = components * (f_row * width + f_col);
	float gray = (0.299f - 0.01f) * (float)d_imgIn[offset] + (0.587f - 0.01f) * (float)d_imgIn[offset + 1] + (0.144f - 0.01f) * (float)d_imgIn[offset + 2];
	d_imgOut[offset] = (unsigned char)gray;
	d_imgOut[offset + 1] = (unsigned char)gray;
	d_imgOut[offset + 2] = (unsigned char)gray;
}

__global__ void RecDecayKernel_Abandon(unsigned char* d_imgIn, unsigned char* d_imgOut, int width, int height, int components, int d, int n) {


	int frag_x = blockIdx.x * blockDim.x + threadIdx.x;
	int frag_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (frag_x >= width || frag_y >= height)
	{
		return;
	}

	int fragInd = components * (frag_y * width + frag_x);
	float recR = 0.0; float recG = 0.0; float recB = 0.0;
	float coeffSum = 0.0;

	for (int off_y = -d; off_y <= d; off_y++) {
		for (int off_x = -d; off_x <= d; off_x++) {
			//
			if ((frag_x + off_x) < 0 || (frag_x + off_x) > width - 1 || (frag_y + off_y) < 0 || (frag_y + off_y) > height - 1) continue;
			//relative offset to the current index
			int offsetInd = components * ((frag_y + off_y) * width + (frag_x + off_x));

			float trigCoeff = decayFunc::recFunc(sqrtf(off_y * off_y + off_x * off_x));
			recR += (float)d_imgIn[offsetInd] * trigCoeff;
			recG += (float)d_imgIn[offsetInd + 1] * trigCoeff;
			recB += (float)d_imgIn[offsetInd + 2] * trigCoeff;
			coeffSum += trigCoeff;
		}
	}
	d_imgOut[fragInd] = recR / coeffSum;
	d_imgOut[fragInd + 1] = recG / coeffSum;
	d_imgOut[fragInd + 2] = recB / coeffSum;

}

__global__ void TrigDecayKernel_Abandon(unsigned char* d_imgIn, unsigned char* d_imgOut, int width, int height, int components, int d) {

	int frag_x = blockIdx.x * blockDim.x + threadIdx.x;
	int frag_y = blockIdx.y * blockDim.y + threadIdx.y;
	int fragInd = components * (frag_y * width + frag_x);
	float trigR(0), trigG(0), trigB(0), coeffSum(0);

	for (int off_y = -d; off_y <= d; off_y++) {
		for (int off_x = -d; off_x <= d; off_x++) {
			//
			if ((frag_x + off_x) < 0 || (frag_x + off_x) > width - 1 || (frag_y + off_y) < 0 || (frag_y + off_y) > height - 1) continue;
			//relative offset to the current index
			int offsetInd = components * ((frag_y + off_y) * width + (frag_x + off_x));

			float trigCoeff = decayFunc::trigFunc(sqrtf(off_y * off_y + off_x * off_x));
			trigR += (float)d_imgIn[offsetInd] * trigCoeff;
			trigG += (float)d_imgIn[offsetInd + 1] * trigCoeff;
			trigB += (float)d_imgIn[offsetInd + 2] * trigCoeff;
			coeffSum += trigCoeff;
		}
	}

	d_imgOut[fragInd] = trigR / coeffSum;
	d_imgOut[fragInd + 1] = trigG / coeffSum;
	d_imgOut[fragInd + 2] = trigB / coeffSum;
}

__global__ void GausDecayKernel_Abandon(unsigned char* d_imgIn, unsigned char* d_imgOut, int width, int height, int components, int d) {

	int frag_x = blockIdx.x * blockDim.x + threadIdx.x;
	int frag_y = blockIdx.y * blockDim.y + threadIdx.y;
	int frag_ind = components * (frag_y * width + frag_x);
	const float dist_adjustment = 1;
	float gausR_temp = 0;
	float gausG_temp = 0;
	float gausB_temp = 0;
	float coefSum = 0;
	//d*d kernel
	for (int off_y = -d; off_y <= d; off_y++) {
		for (int off_x = -d; off_x <= d; off_x++) {
			//test if out of range
			if ((frag_x + off_x) < 0 || (frag_x + off_x) > width - 1 || (frag_y + off_y) < 0 || (frag_y + off_y) > height - 1) continue;
			//relative offset to the current index
			int offsetInd = components * ((frag_y + off_y) * width + (frag_x + off_x));
			float gaus = decayFunc::gausFunc(dist_adjustment * sqrtf(off_y * off_y + off_x * off_x));
			gausR_temp += (float)d_imgIn[offsetInd] * gaus;
			gausG_temp += (float)d_imgIn[offsetInd + 1] * gaus;
			gausB_temp += (float)d_imgIn[offsetInd + 2] * gaus;
			coefSum += gaus;
		}
	}

	d_imgOut[frag_ind] = gausR_temp / coefSum;
	d_imgOut[frag_ind + 1] = gausG_temp / coefSum;
	d_imgOut[frag_ind + 2] = gausB_temp / coefSum;
}


//decayFunc::decayFunc(const char* imgAdr) {
//
//
//
//}