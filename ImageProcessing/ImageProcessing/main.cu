
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
#include <vector>

using namespace std;


unsigned char* h_imgIn, * h_imgOut;

//const int decayRange = 15;
const int imageCount = 28;
const int numOfImgToProcess = 28;
const int numOfImgInCirclBuff = 5;
const int numOfImgCtrdg = 5;

__device__ float gausFunc(float x) {

	float sig = 2.0f, pi = 3.1415926f, niu = 0.0f;
	//gaussian equation
	return 1 / (sig * powf(2 * pi, 0.5)) * expf(-0.5 * ((x - niu) / sig) * ((x - niu) / sig));
}

__device__ float recFunc(float x) {

	float d = 1;
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

void printResult(vector<double>* time, int imgW, int imgH, int channel) {

	vector<double>::iterator iter;
	int c = 0;
	double t = 0.0;
	float average = 0.0;
	float stDev = 0.0;

	for (iter = time->begin(); iter != time->end(); iter++) {
		c++;
		t += *iter;
	}
	average = (float)t / c;

	float temp = 0.0;
	for (iter = time->begin(); iter != time->end(); iter++) {
		temp += (*iter - average) * (*iter - average);
	}

	stDev = sqrt(temp / c);

	cout << "\naverage time per loop: " << average << "s +- " << stDev<< "s"<< endl;
	cout << "image size (in byte) = " << imgW << " * " << imgH << " * " << channel << " = " << imgW * imgH * channel << endl;

}

//5 in 1 out
__global__ void RecDecayKernel_ctrdg(unsigned char* d_imgIn, unsigned char* d_imgOut, int width, int height, int components, int n_Img) {

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
	float adjust = 0.1;

	for (int i = n_Img -1 ; i >= 0; i--)
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
cudaError_t ImgProcCUDA(unsigned char** h_imgIn, unsigned char* h_imgOut, int* width, int* height, int components, int nImgIn) {
	unsigned char* d_imgIn = 0;
	unsigned char* d_imgOut = 0;
	cudaError_t cudaStatus;

	unsigned int imgSize = *width * *height * components;
	unsigned int buffSize = imgSize * nImgIn;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!\nDo you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//bufferSize = width * height * n_image( = 5)
	cudaStatus = cudaMalloc((void**)&d_imgIn, buffSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_imgOut, imgSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//TODO: copy the latest imageIn to the memory on GPU with order
	//cudaStatus = cudaMemcpy(d_imgIn + nImgIn % 5 * imageSize, h_imgIn[i], imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
	//free the d_imgIn until finish reading
	for (int i = 0; i < nImgIn; i++)
	{
		cudaStatus = cudaMemcpy(d_imgIn + i * imgSize, h_imgIn[i], imgSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}

	// Launch a kernel on the GPU.
	const int TILE = 16;
	dim3 dimGrid(ceil((float)*width / TILE), ceil((float)*height / TILE));
	dim3 dimBlock(TILE, TILE, 1);

	RecDecayKernel_ctrdg <<<dimGrid, dimBlock>>> (d_imgIn, d_imgOut, *width, *height, components, nImgIn);

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
	cudaStatus = cudaMemcpy(h_imgOut, d_imgOut, imgSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_imgIn);
	cudaFree(d_imgOut);

	return cudaStatus;
}


cudaError_t ImgProcCUDA_cartridgeBuff(int numOfImgToProc, int numImages) {

	cudaError_t cudaStatus;

	int width, height;
	unsigned long int imageSize = 0;
	unsigned long int bufferSize = 0;

	int components = 0;
	int requiredComponents = 3;

	unsigned char* h_imgInCtrdgBuff[numOfImgCtrdg] = { nullptr };

	//time
	clock_t tStart;
	vector<double> timeRecord;

	cout << "\nReading input image";

	//======================================================
	//read 5 images to the cartridge buffer
	//get width & height
	for (int i = 0; i < numOfImgCtrdg; i++)
	{
		string fileName = "images/source(" + to_string(i) + ").jpg";
		h_imgInCtrdgBuff[i] = stbi_load(fileName.c_str(), &(width), &(height), &components, requiredComponents);
		if (!h_imgInCtrdgBuff[i]) {
			cout << "Cannot read input image, invalid path?" << endl;
			exit(-1);
		}
	}

	imageSize = width * height * components;
	bufferSize = imageSize * numOfImgCtrdg;

	//prepare output memory to be copied into
	h_imgOut = (unsigned char*)malloc(imageSize * sizeof(unsigned char));
	if (h_imgOut == NULL) {
		cout << "host side malloc failed" << endl;
		exit(-1);
	}

	//======================================================
	//start from 5th
	for (int i = numOfImgCtrdg; i < numOfImgToProc; i++)
	{
		//========================================
		//measure performance
		tStart = clock();

		cout << "\nProcessing the image";
		// Run the memory copies, kernel, copy back from the helper
		cudaStatus = ImgProcCUDA(h_imgInCtrdgBuff, h_imgOut, &width, &height, components, numOfImgCtrdg);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Cuda krenel failed!");
			exit(-1);
		}

		cout << "\nTime taken this loop: " << (double)(clock() - tStart) / CLOCKS_PER_SEC << endl;

		cout << "\nSaving the image";
		//save the output image
		string outputFileName = "images/zresult-" + to_string(i - 5) + ".png";
		int result = stbi_write_png(outputFileName.c_str(), width, height, components, h_imgOut, 0);
		if (!result) {
			cout << "Something went wrong during writing. Invalid path?" << endl;
			exit(-1);
		}

		stbi_image_free(h_imgInCtrdgBuff[0]);

		//move each img data 1 slot forward
		for (int j = 0; j < numOfImgCtrdg - 1; j++)
		{
			h_imgInCtrdgBuff[j] = h_imgInCtrdgBuff[j + 1];
		}

		//always write to the final slot
		string fileName = "images/source(" + to_string(i % numImages) + ").jpg";
		h_imgInCtrdgBuff[4] = stbi_load(fileName.c_str(), &(width), &(height), &components, requiredComponents);
		if (!h_imgInCtrdgBuff[4]) {
			cout << "Cannot read input image, invalid path?" << endl;
			exit(-1);
		}

		//output time taken each loop
		
		timeRecord.push_back((double)(clock() - tStart) / CLOCKS_PER_SEC);
	}

	printResult(&timeRecord, width, height, components);

	return cudaStatus;
}



__global__ void RecDecayKernel_circularBuff(unsigned char* d_imgIn, unsigned char* d_imgOut, int width, int height, int components, int ith_imgIn) {

	int frg_x = blockIdx.x * blockDim.x + threadIdx.x;
	int frg_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (frg_x >= width || frg_y >= height) return;

	int frgInd = components * (frg_y * width + frg_x);
	unsigned long int imagSize = width * height * components;

	d_imgOut[frgInd] = 0;
	d_imgOut[frgInd + 1] = 0;
	d_imgOut[frgInd + 2] = 0;

	float _r = 0; float _g = 0; float _b = 0;
	float coeffSm = 0.0;
	float adjst = 1.0;

	for (int c = 4; c >= 0; c--) 
	{
		//decay
		coeffSm += gausFunc(adjst * (float)(4 - c));
		//_r += recFunc(adjst * (float)c) * d_imgIn[frgInd + imagSize * ((ith_imgIn + c + 1) % 5)];
		//_g += recFunc(adjst * (float)c) * d_imgIn[frgInd + imagSize * ((ith_imgIn + c + 1) % 5) + 1];
		//_b += recFunc(adjst * (float)c) * d_imgIn[frgInd + imagSize * ((ith_imgIn + c + 1) % 5) + 2];

		_r += gausFunc(adjst * (float)(4 - c)) * d_imgIn[frgInd + imagSize * ((ith_imgIn + c - 4) % 5)];
		_g += gausFunc(adjst * (float)(4 - c)) * d_imgIn[frgInd + imagSize * ((ith_imgIn + c - 4) % 5) + 1];
		_b += gausFunc(adjst * (float)(4 - c)) * d_imgIn[frgInd + imagSize * ((ith_imgIn + c - 4) % 5) + 2];
	}																		 

	d_imgOut[frgInd + 0] = _r / coeffSm;
	d_imgOut[frgInd + 1] = _g / coeffSm;
	d_imgOut[frgInd + 2] = _b / coeffSm;
}

cudaError_t ImgProcCUDA_CircularBuff(int numOfImgToProc, int numImages) {


	//prepare variables
	//device
	unsigned char* d_imgIn_test = 0;
	unsigned char* d_imgOut_test = 0;
	cudaError_t cudaStatus;
	//read image
	int components_test = 0;
	int requiredComponents_test = 3;
	int w_test, h_test;
	unsigned char* h_imgIn_test, * h_imgOut_test;

	//time
	clock_t tStart;
	vector<double> timeRecord;

	//read 1 image to fetch some variables
	h_imgIn_test = stbi_load("images/source(0).jpg", &(w_test), &(h_test), &components_test, requiredComponents_test);
	if (!h_imgIn_test) {
		cout << "Cannot read input image, invalid path?" << endl;
		exit(-1);
	}

	unsigned int imgSize_test = w_test * h_test * components_test;
	unsigned long int circularBufferSize = imgSize_test * numOfImgInCirclBuff;

	h_imgOut_test = (unsigned char*)malloc(imgSize_test * sizeof(unsigned char));
	if (h_imgOut_test == NULL) {
		cout << "host side malloc failed" << endl;
		exit(-1);
	}

	string outImgName = "0";

	//=========================================================
	//allcate first, just for once
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!\nDo you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//allocate a memory with size of 5 images, preparing circular buffer
	cudaStatus = cudaMalloc((void**)&d_imgIn_test, circularBufferSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//1 image out after the kernel operation
	cudaStatus = cudaMalloc((void**)&d_imgOut_test, imgSize_test * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//=========================================================
	//copy 4 img to GPU at beginnning
	for (int j = 0; j < 4; j++) {

		string fileName = "images/source(" + to_string(j) + ").jpg";
		h_imgIn_test = stbi_load(fileName.c_str(), &(w_test), &(h_test), &components_test, requiredComponents_test);

		cudaStatus = cudaMemcpy(d_imgIn_test + j * imgSize_test, h_imgIn_test, imgSize_test * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed 0!");
			goto Error;
		}

		stbi_image_free(h_imgIn_test);
	}

	//=========================================================
	//from the 5th image
	for (int ithImg = 4; ithImg < numOfImgToProc; ithImg++)
	{
	
		std::cout << "reading image" << std::endl;
		//=========================================================
		//READ
		string fileName = "images/source(" + to_string(ithImg % numImages) + ").jpg";
		h_imgIn_test = stbi_load(fileName.c_str(), &(w_test), &(h_test), &components_test, requiredComponents_test);
		if (!h_imgIn_test) {
			cout << "Cannot read input image, invalid path?" << endl;
			exit(-1);
		}

		tStart = clock();
		//=====================================
		//Copy
		//Copy 1 image at a time using circular buffer
		//TODO: copy the latest imageIn to the memory on GPU with order
		//cudaStatus = cudaMemcpy(d_imgIn + nImgIn % 5 * imageSize, h_imgIn[i], imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
		//free the d_imgIn until finish reading
		cudaStatus = cudaMemcpy(d_imgIn_test + (ithImg % numOfImgInCirclBuff) * imgSize_test, h_imgIn_test, imgSize_test * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed 2 !");
			goto Error;
		}

		//=======================================================
		//free loaded image from memory
		delete[] (h_imgIn_test);
	

		//=======================================================
		//Kernel
		//taking 5 pictures
		// Launch a kernel on the GPU.
		//prepare kernel
		const int TILE = 16;
		dim3 dimGrid(ceil((float)w_test / TILE), ceil((float)h_test / TILE));
		dim3 dimBlock(TILE, TILE, 1);
		
		cudaEvent_t startT, stopT;
		float time;
		cudaEventCreate(&startT, 0);
		cudaEventRecord(startT);

		RecDecayKernel_circularBuff <<< dimGrid, dimBlock >>> (d_imgIn_test, d_imgOut_test, w_test, h_test, components_test, ithImg);

		cudaEventCreate(&stopT);
		cudaEventRecord(stopT, 0);
		cudaEventSynchronize(stopT);
		cudaEventElapsedTime(&time, startT, stopT);
		cudaEventDestroy(startT);
		cudaEventDestroy(stopT);
		std::cout << "kernel excecution time: " << time * 0.001f << "s\n";
		

		//========================================================
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
			goto Error;
		}

		// Copy from dev to host .
		cudaStatus = cudaMemcpy(h_imgOut_test, d_imgOut_test, w_test*h_test*components_test * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed 1!");
			goto Error;
		}

		cout << "\nTime taken this loop: " << (double)(clock() - tStart) / CLOCKS_PER_SEC << endl;

		//write into PNG=============================================
		cout << "Saving the image\n";
		outImgName = "images/zresult-" + to_string(ithImg - 4) + ".png";
		int result = stbi_write_png(outImgName.c_str(), w_test, h_test, components_test, h_imgOut_test, 0);
		if (!result) {
			cout << "Something went wrong during writing. Invalid path?" << endl;
			//return 0;
		}


		//output time taken each loop
		timeRecord.push_back((double)(clock() - tStart) / CLOCKS_PER_SEC);
	}

	printResult(&timeRecord, w_test, h_test, components_test);

	//===========================================================
	//free memory on GPU at last
Error:
	cudaFree(d_imgIn_test);
	cudaFree(d_imgOut_test);

	return cudaStatus;
}



int main()
{
	cudaError_t cudaStatus;

	ImgProcCUDA_CircularBuff(300, imageCount);
	//ImgProcCUDA_cartridgeBuff(200, imageCount);
	

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

