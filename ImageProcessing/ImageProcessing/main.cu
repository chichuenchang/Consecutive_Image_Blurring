
#include "decayFunc.h"

#pragma warning(disable:4996)
#define _USE_MATH_DEFINES
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
//#include <string>

using namespace std;

int width, height;
unsigned long int imageSize = 0;
unsigned long int bufferSize = 0;
unsigned char* h_imgIn, * h_imgOut;

const int n = 5;
const int imageCount = 28;
const int decayRange = 15;

unsigned char* h_imgIns[n] = { nullptr };

__device__ float gausFunc(float x) {

	float sig = 2.0f, pi = 3.1415926f, niu = 0.0f;
	//gaussian equation
	return 1 / (sig * powf(2 * pi, 0.5)) * expf(-0.5 * ((x - niu) / sig) * ((x - niu) / sig));
}

__device__ float recFunc(float x) {

	float d = 5;
	if (x < d) return 1.0;
	else if (x == d) return 0.5;
	else return 0;
}

__device__ float trigFunc(float x) {

	const float k = 0.3;
	float y = 1 - k * fabs(x);

	if (y < 0) y = 0;
	return y;
}

//5 in 1 out
__global__ void RecDecayKernel(unsigned char* d_imgIn, unsigned char* d_imgOut, int width, int height, int components, int n) {

	int frag_x = blockIdx.x * blockDim.x + threadIdx.x;
	int frag_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (frag_x >= width || frag_y >= height) return;

	int fragInd = components * (frag_y * width + frag_x);
	unsigned long int imgSize = width * height * components;

	d_imgOut[fragInd] = 0;
	d_imgOut[fragInd + 1] = 0;
	d_imgOut[fragInd + 2] = 0;

	float r = 0; float g = 0; float b = 0;
	float coeffSum = 0.0;
	float adjust = 1;

	for (int i = 0; i < n; i++)
	{
		//decay
		coeffSum += recFunc(adjust * (float)i);
		r += recFunc(adjust * (float)i) * d_imgIn[fragInd + imgSize * i];
		g += recFunc(adjust * (float)i) * d_imgIn[fragInd + imgSize * i + 1];
		b += recFunc(adjust * (float)i) * d_imgIn[fragInd + imgSize * i + 2];
	}

	d_imgOut[fragInd + 0] = r / coeffSum;
	d_imgOut[fragInd + 1] = g / coeffSum;
	d_imgOut[fragInd + 2] = b / coeffSum;
}


// 5 in 1 out
cudaError_t ImgProcCUDA(unsigned char** h_imgIn, unsigned char* h_imgOut, int* width, int* height, int components, int overrideN) {
	unsigned char* d_imgIn = 0;
	unsigned char* d_imgOut = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!\nDo you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&d_imgIn, bufferSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_imgOut, imageSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input image from host memory to GPU buffers.
	for (int i = 0; i < overrideN; i++)
	{
		cudaStatus = cudaMemcpy(d_imgIn + i * imageSize, h_imgIn[i], imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}

	// Launch a kernel on the GPU.
	const int TILE = 16;
	dim3 dimGrid(ceil((float)*width / TILE), ceil((float)*height / TILE));
	dim3 dimBlock(TILE, TILE, 1);

	RecDecayKernel << <dimGrid, dimBlock >> > (d_imgIn, d_imgOut, *width, *height, components, overrideN);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output image from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(h_imgOut, d_imgOut, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_imgIn);
	cudaFree(d_imgOut);

	return cudaStatus;
}


//TODO: 1 new keeps coming in, copy to the oldest memory on GPU using mod, 1 out


int main()
{
	//read the image first
	int components = 0;
	int requiredComponents = 3;


	//this loads the file, returns its resultion and number of componnents
	cout << "\nReading input image";
	//string a = "images/WeChat Image_201912132356291.jpg";
	//h_imgIn = stbi_load(a.c_str(), &(width), &(height), &components, requiredComponents);

	for (int i = 0; i < n; i++)
	{
		string fileName = "images/(" + to_string(i) + ").jpg";
		h_imgIns[i] = stbi_load(fileName.c_str(), &(width), &(height), &components, requiredComponents);
	}

	/*if (!h_imgIn) {*/
	for (int i = 0; i < n; i++)
	{
		if (!h_imgIns[i]) {
			cout << "Cannot read input image, invalid path?" << endl;
			exit(-1);
		}
	}

	imageSize = width * height * components;
	bufferSize = imageSize * n;
	h_imgOut = (unsigned char*)malloc(imageSize * sizeof(unsigned char));
	if (h_imgOut == NULL) {
		cout << "malloc failed" << endl;
		exit(-1);
	}

	cudaError_t cudaStatus;
	for (int i = 0; i < n - 1; i++)
	{
		cout << "\nProcessing the image";

		cudaStatus = ImgProcCUDA(h_imgIns, h_imgOut, &width, &height, components, i + 1);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Cuda krenel failed!");
			return 1;
		}
		cout << "\nSaving the image";
		//save the output image
		string outputFileName = "images/result-" + to_string(i) + ".png";
		int result = stbi_write_png(outputFileName.c_str(), width, height, components, h_imgOut, 0);
		if (!result) {
			cout << "Something went wrong during writing. Invalid path?" << endl;
			return 0;
		}
	}

	for (int i = n; i < imageCount + 1; i++)
	{
		cout << "\nProcessing the image";
		// Run the memory copies, kernel, copy back from the helper
		//cudaError_t cudaStatus = ImgProcCUDA(h_imgIn, h_imgOut, &width, &height, components, decayRange);
		cudaStatus = ImgProcCUDA(h_imgIns, h_imgOut, &width, &height, components, n);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Cuda krenel failed!");
			return 1;
		}
		cout << "\nSaving the image";
		//save the output image
		string outputFileName = "images/result-" + to_string(i - 1) + ".png";
		int result = stbi_write_png(outputFileName.c_str(), width, height, components, h_imgOut, 0);
		if (!result) {
			cout << "Something went wrong during writing. Invalid path?" << endl;
			return 0;
		}

		if (i < imageCount)
		{
			stbi_image_free(h_imgIns[0]);
			for (int j = 0; j < n - 1; j++)
			{
				h_imgIns[j] = h_imgIns[j + 1];
			}

			string fileName = "images/(" + to_string(i) + ").jpg";
			h_imgIns[4] = stbi_load(fileName.c_str(), &(width), &(height), &components, requiredComponents);
			if (!h_imgIns[4]) {
				cout << "Cannot read input image, invalid path?" << endl;
				exit(-1);
			}
		}
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	cout << "\nDone\n";
	return 0;
}

