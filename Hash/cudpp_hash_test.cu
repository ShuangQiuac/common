/*
 * cudpp_hash_test.cu
 *
 *  Created on: 2019-2-21
 *      Author: qiushuang
 */
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cudpp_hash.h"
#include "type.h"
#include "utility.h"

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define CHECK_PTR_RETURN(ptr, ...) {									\
	if (ptr == NULL) {												\
		printf (__VA_ARGS__);										\
		printf ("Error in returned value: NULL\n");					\
		exit (1);													\
	} }

#define CHECK_KERNEL_LAUNCH() {\
		if ( cudaSuccess != cudaGetLastError() ) {\
		    printf( "Error in lauching kernel!\n" );\
			exit(1); }}

void test_driver (void)
{
	uint N = 1UL << 24;
	uint* key = (uint*) malloc (sizeof(uint) * N);
	CHECK_PTR_RETURN (key, "malloc host keys error!\n");
	uint* value = (uint*) malloc (sizeof(uint) * N);
	CHECK_PTR_RETURN (value, "malloc host values error!\n");
	uint* input = (uint*) malloc (sizeof(uint) * N);
	CHECK_PTR_RETURN (input, "malloc input on host error!\n");
	uint* output = (uint*) malloc (sizeof(uint) * N);
	CHECK_PTR_RETURN (output, "malloc output on host error!\n");

	for (uint i=0; i<N; i++)
	{
		key[i] = i+1;
		value[i] = key[i];
		input[i] = N-i;
	}
	uint*d_key, *d_value;
	CUDA_CHECK_RETURN(cudaMalloc (&d_key, sizeof(uint) * N));
	CUDA_CHECK_RETURN(cudaMalloc (&d_value, sizeof(uint) * N));
	CUDA_CHECK_RETURN(cudaMemcpy(d_key, key, sizeof(uint) * N, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_value, value, sizeof(uint) * N, cudaMemcpyHostToDevice));
	uint*d_input, *d_output;
	CUDA_CHECK_RETURN(cudaMalloc (&d_input, sizeof(uint) * N));
	CUDA_CHECK_RETURN(cudaMalloc (&d_output, sizeof(uint) * N));
	CUDA_CHECK_RETURN(cudaMemcpy(d_input, input, sizeof(uint) * N, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemset(d_output, 0, sizeof(uint) * N));

	CUDPPHashTableConfig hashtabConfig = {CUDPP_BASIC_HASH_TABLE, (uint)N, 1.5};
	CUDPPHandle cudppHandle;
	cudppCreate(&cudppHandle);
	printf ("cudppHandle value: %lu\n", cudppHandle);
	CUDPPHandle tableHandle;
	cudppHashTable(cudppHandle, &tableHandle, &hashtabConfig);
	printf ("cudppHashHandle value: %lu\n", tableHandle);

	evaltime_t start, end;
	float hashInsertTime = 0;
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
	cudppHashInsert(tableHandle, (void*)d_key, (void*)d_value, N);
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	hashInsertTime = (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	printf ("Hash insert  time: %f\n", hashInsertTime);
	cudppHashRetrieve(tableHandle, d_input, d_output, N);

	CUDA_CHECK_RETURN (cudaMemcpy(output, d_output, sizeof(uint) * N, cudaMemcpyDeviceToHost));

	for (uint i=0; i<N; i++)
	{
		printf ("key: %u, value: %u\t\tinput key: %u, retrieved value: %u\n", key[i], value[i], input[i], output[i]);
	}

	free(input);
	free(output);
	free(key);
	free(value);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_key);
	cudaFree(d_value);
}
