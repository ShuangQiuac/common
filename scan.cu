/*
 * scan.cu
 *
 *  Created on: 2018-9-2
 *      Author: qiushuang
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utility.h"

#define THREADS_PER_SCAN 1024
#define NUM_WARPS_PER_BLOCK 32
#define WARP_WIDTH 32
#define MAX_BLOCK_SIZE 1024

template <typename T>
__global__ void local_scan (T * data, T * block_sum)
{
	extern __shared__ T temp[];
	int tid = threadIdx.x;
	temp[tid] = data[tid + blockIdx.x * blockDim.x];

	for (int i=1; i<blockDim.x; i<<=1)
	{
		__syncthreads();
		T temp2 = (tid>=i)? temp[tid-i]:0;
		__syncthreads();
		temp[tid] += temp2;
	}
	__syncthreads();

	// write partial block sum to block_sum array
	if (tid == blockDim.x - 1)
		block_sum[blockIdx.x] = temp[blockDim.x - 1];
}

template <typename T>
__global__ void
scan_within_block (T * array, uint num, int bulk)
{
	uint gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint tid = threadIdx.x;
	uint warpid = threadIdx.x / WARP_WIDTH;
	uint laneid = threadIdx.x % WARP_WIDTH;

	__shared__ __align__(sizeof(T)) unsigned char mysumpb[NUM_WARPS_PER_BLOCK * sizeof(T)];
	T * sumpb = reinterpret_cast<T *> (mysumpb);

	__shared__ __align__(sizeof(T)) unsigned char mypartial[THREADS_PER_SCAN * sizeof(T)];
	T * partial = reinterpret_cast<T *> (mypartial);

	__align__(sizeof(T)) unsigned char mysumbs[MAX_BLOCK_SIZE * sizeof(T)];
	T * sumbs = reinterpret_cast<T *> (mysumbs);

	if (gid * bulk >= num)
		return;

	int i;
	for (i = 1; i < bulk; i++)
	{
		array[gid * bulk + i] += array[gid * bulk + i - 1];
	}
	partial[tid] = array[gid * bulk + bulk - 1];
	__syncthreads();

	T x = partial[tid];
	/* scan per warp */
#pragma unroll
	for (i = 1; i < WARP_WIDTH; i <<= 1)
	{
		T y = __shfl_up_sync (x, i, WARP_WIDTH);
		if (laneid >= i)
			x += y;
	}
	partial[tid] = x;
	/* sums per block */
	if (laneid == WARP_WIDTH - 1)
	{
		sumpb[warpid] = x;
	}
	__syncthreads();

	/* scan of sumbp */
	if (warpid == 0)
	{
		x = sumpb[tid];
#pragma unroll
		for (i = 1; i < WARP_WIDTH; i <<= 1)
		{
			T y = __shfl_up_sync (x, i, WARP_WIDTH);
			if (laneid >= i)
				x += y;
		}
		sumpb[laneid] = x;
	}
	__syncthreads();

	if (warpid > 0)
	{
		partial[tid] += sumpb[warpid - 1];
	}
	__syncthreads();
	if (tid > 0)
	{
		for (i = 0; i < bulk; i++)
		{
			array[gid * bulk + i] += partial[tid - 1];
		}
	}
	if (tid == 0)
	{
		sumbs[blockIdx.x] = sumpb[WARP_WIDTH - 1];
	}
}

template <typename T>
__global__ void
scan_block_sum (void)
{
	uint gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint tid = threadIdx.x;
	uint warpid = threadIdx.x / WARP_WIDTH;
	uint laneid = threadIdx.x % WARP_WIDTH;

	__shared__ __align__(sizeof(T)) unsigned char mysumpb[NUM_WARPS_PER_BLOCK * sizeof(T)];
	T * sumpb = reinterpret_cast<T *> (mysumpb);

	__align__(sizeof(T)) unsigned char mysumbs[MAX_BLOCK_SIZE * sizeof(T)];
	T * sumbs = reinterpret_cast<T *> (mysumbs);

	T x;
	int i;
	if (gid < MAX_BLOCK_SIZE)
	{
		x = sumbs[gid];
#pragma unroll
		for (i = 1; i < WARP_WIDTH; i <<= 1)
		{
			T y = __shfl_up_sync (x, i, WARP_WIDTH);
			if (laneid >= i)
				x += y;
		}
		sumbs[gid] = x;
		if (laneid == WARP_WIDTH - 1)
		{
			sumpb[warpid] = x;
		}
		__syncthreads();

		/* scan of sumbp */
		if (warpid == 0)
		{
			x = sumpb[tid];
#pragma unroll
			for (i = 1; i < WARP_WIDTH; i <<= 1)
			{
				T y = __shfl_up_sync (x, i, WARP_WIDTH);
				if (laneid >= i)
					x += y;
			}
			sumpb[laneid] = x;
		}
		__syncthreads();

		if (warpid > 0)
		{
			sumbs[gid] += sumpb[warpid - 1];
		}
	}
}

template <typename T>
__global__ void
scan_blocks (T * array, uint num, int bulk)
{
	uint gid = blockIdx.x * blockDim.x + threadIdx.x;

	__align__(sizeof(T)) unsigned char mysumbs[MAX_BLOCK_SIZE * sizeof(T)];
	T * sumbs = reinterpret_cast<T *> (mysumbs);

	if (gid * bulk >= num)
		return;
	if (blockIdx.x > 0)
	{
		int i;
		for (i = 0; i < bulk; i++)
		{
			array[gid * bulk + i] += sumbs[blockIdx.x - 1];
		}
	}
}

template <typename T>
void inclusive_scan (T * data, uint num, cudaStream_t stream)
{
	int bulk = (num + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
	bulk = (bulk + THREADS_PER_SCAN - 1) / THREADS_PER_SCAN;
	int block_size = MAX_BLOCK_SIZE;
	scan_within_block <<<block_size, THREADS_PER_SCAN, NUM_WARPS_PER_BLOCK+THREADS_PER_SCAN, stream>>> (data, num, bulk);
//	scan_block_sum <<<1, THREADS_PER_SCAN, NUM_WARPS_PER_BLOCK+THREADS_PER_SCAN, stream>>> ();
//	scan_blocks <<<block_size, THREADS_PER_SCAN, NUM_WARPS_PER_BLOCK+THREADS_PER_SCAN, stream>>> (data, num, bulk);
}

int main (void)
{
	int i;
	int n;
	n = 4096 * 256 + 1;
	unsigned int * data = (unsigned int *) malloc (sizeof(unsigned int) * (n+1));
	CHECK_PTR_RETURN (data, "malloc scan data error!\n");
	for (i = 0; i < n; i++)
	{
		data[i] = 1;
	}
	unsigned int * d_data;
	CUDA_CHECK_RETURN (cudaMalloc (&d_data, sizeof(unsigned int) * (n+1)));
	CUDA_CHECK_RETURN (cudaMemcpy (d_data, data, sizeof(unsigned int) * (n+1), cudaMemcpyHostToDevice));

	inclusive_scan <unsigned int> (d_data, n + 1, NULL);
	CUDA_CHECK_RETURN (cudaMemcpy (data, d_data, sizeof(unsigned int) * (n+1), cudaMemcpyDeviceToHost));
	printf ("data[%u]=%d, data[%u] = %u\tdata[%u] = %u, data[%u] = %d\n", 0, data[0], n-1, data[n - 1], n-2, data[n - 2], n, data[n]);

	cudaFree (d_data);
	free (data);
	return 0;
}
