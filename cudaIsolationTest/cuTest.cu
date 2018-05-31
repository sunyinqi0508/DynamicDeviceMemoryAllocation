#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>
#include "cuTest.cuh"
template<class _Ty>
__device__ __host__ __forceinline__ 
int binarySearch(const _Ty* __restrict__ orderedList, int lowerbound, int upperbound, const _Ty& key) {
	while (upperbound > lowerbound) {
		int mid = (lowerbound + upperbound) >> 1;
		if (mid == lowerbound) {
			return orderedList[mid] == key ? lowerbound : -lowerbound;
		}
		else {
			if (orderedList[mid] > key)
				upperbound = mid;
			else if (orderedList[mid] < key)
				lowerbound = mid;
			else
				return mid;
		}
	}
	return orderedList[lowerbound] == key ? lowerbound : -lowerbound;
}
template<class _Ty>
inline int my_lower_bound(const _Ty *orderedList, int _First, const int _Last,
		const _Ty& _Val)
{	
	int _Count = _Last - _First;
	int _LCount = 0;
	while (0 < _Count)
	{	
		int _Count2 = _Count >> 1; 

		if (orderedList[_LCount + _Count2] < _Val)
		{	
			_LCount += _Count2 + 1;
			_Count -= _Count2 + 1;
		}
		else
			_Count = _Count2;
	}

	return _LCount;
}


__global__ 
void cutest(Bridge* b) {
	Bridge *c = new Bridge();
	unsigned long long idx = threadIdx.x + blockDim.x * blockIdx.x;
	curandDirectionVectors64_t dir;
	atomicAdd(&(b->a), threadIdx.x);
	curandStateMRG32k3a_t state;
	
	curand_init(1234,idx*idx, idx, &state);
	float this_rand = curand_uniform(&state);
	b->b = c->b ? c->b:this_rand;
#ifdef __CUDACC__
	//printf("hello 23 from %d %d %d %f\n", blockIdx.x, threadIdx.x, _MSVC_LANG, this_rand);
#endif
}

void cudasync() {
	cudaDeviceSynchronize();
}
void init() {
#ifndef __CUDACC__
//#if __cplusplus > 201403
	msvcInit();
	printf("%ld %ld", __cplusplus, _MSVC_LANG);
#endif
	cudaSetDevice(0);
	int* devcount = new int(-1);
	int* currDevice = new int(-1); 
	size_t heap_size;
	cudaLimit::cudaLimitMallocHeapSize;
	cudaGetDeviceCount(devcount);
	cudaDeviceProp* prop = (cudaDeviceProp *)malloc(sizeof(cudaDeviceProp)),
				  *prop2 = (cudaDeviceProp *)malloc(sizeof(cudaDeviceProp));

	cudaGetDeviceProperties(prop, 0);
	cudaGetDeviceProperties(prop2, 1);

	cudaGetDevice(currDevice);
	cudaDeviceGetLimit(&heap_size, cudaLimit::cudaLimitMallocHeapSize); 

	printf("Device count: %d\nProperties 1: %s\nProperties 2: %s\n\
Current Device: %d\n"
		, *devcount, prop->name, prop2->name, *currDevice);
	std::cout << "Total Memory: " << prop->totalGlobalMem << '\n';
	std::cout << "Total Memory: " << prop->totalConstMem << '\n';
	std::cout<<"Heap Size: "<<heap_size<<'\n';

}

struct CheckTable {
	bool _0 : 1;
	bool _1 : 1;
	bool _2 : 1;
	bool _3 : 1;
	bool _4 : 1;
	bool _5 : 1;
	bool _6 : 1;
	bool _7 : 1;
	bool operator[](const char idx) {
		switch (idx) {
		case 0: return _0;
		case 1: return _1;
		case 2: return _2;
		case 3: return _3;
		case 4: return _4;
		case 5: return _5;
		case 6: return _6;
		case 7: return _7;
		default: return _0;
		}
	}
	bool operator()(const char idx, const bool& val) {
		switch (idx) {
		case 0: _0 = val; break;
		case 1: _1 = val; break;
		case 2: _2 = val; break;
		case 3: _3 = val; break;
		case 4: _4 = val; break; 
		case 5: _5 = val; break;
		case 6: _6 = val; break;
		case 7: _7 = val; break;
		default: _0 = val; 
		}
	}
};
__global__ void memorypool_init(
	int ** d_mp,
	int* current_memoryblock, int* current_offset, curandStateMRG32k3a_t* state
) {
	d_mp[0][0] = 1;
	*current_memoryblock = 0;
	*current_offset = 0;
	curand_init(1234, 954, 465, state);

}
__global__ void parallelized_memory_allocation_test(

	int** memory_pool, int64_t* memory_pool_index, int* current_memoryblock, int* current_offset, int n,
	int* _idxs, const int poolsize, curandStateMRG32k3a_t *state, int* finished

) {

	const int _idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = _idxs[_idx] ? _idxs[_idx] : _idx;//resume last work
	bool overflew = false;
	for (; idx < n; idx += blockDim.x * gridDim.x) {
		
		const int result_count = curand_uniform(state) * 32.f;
		int this_offset = atomicAdd(current_offset, result_count);
		int this_memoryblock = *current_memoryblock;
		int this_end = this_offset + result_count;

		memory_pool_index[idx] =
			(this_memoryblock << 54ll) | (this_offset << 27ll) | result_count;

		if (this_end > poolsize)
		{
			for (; this_offset < poolsize; ++this_offset) {
				memory_pool[this_memoryblock][this_offset] = idx * 10000 + this_offset;
			}
			this_offset = 0;
			this_end -= poolsize;
			++this_memoryblock;
			overflew = true;
		}
		for (; this_offset < this_end; ++this_offset) {
			memory_pool[this_memoryblock][this_offset] = idx * 10000 + this_offset;
		}
		if (overflew)
			break;
	}
	_idxs[_idx] = idx;
	atomicAnd(finished, idx >= n);
}

__global__ void substract(int* current_memoryblock, int* current_offset, const int poolsize) {
	*current_memoryblock++;
	*current_offset -= poolsize;
}
void cutest_launch(int griddim, int blockdim, Bridge *b) {
	printf("launching cuda kernels %d %d\n", griddim, blockdim);
//	b->a = 0;
//	b->b = 0.0f;
	//std::random_device rd{ };
	std::mt19937_64 engine{1234};
	std::uniform_int_distribution<int> uid{};
	int orderedList[128], key = 43;// uid(engine) % 10000;
	for (int i = 0; i < 128; i++)
		orderedList[i] = uid(engine)%100;
	std::sort(orderedList, orderedList + 128);
	for (int i = 0; i < 128; i++)
		printf("%d ", orderedList[i]);
	int result = binarySearch(orderedList, 0, 128, key);
	int result2 = my_lower_bound(orderedList, 0, 127, key);
	int* result3 = std::lower_bound(orderedList, orderedList + 127, key);
	printf("\n\nresult: %d %d %d %d\n", result, result2, *result3, orderedList[abs(result)], key);
	Bridge* dev_b;
	
	void *cumemtest;
	std::numeric_limits<int>::max();
	cudaMalloc(&dev_b, 8Ui64);
	cudaMemcpy(dev_b, b, sizeof(Bridge), cudaMemcpyHostToDevice);
	cutest << <1, 1>> >(dev_b);
	cudaDeviceSynchronize();
	cudaMemcpy(b, dev_b, sizeof(Bridge), cudaMemcpyDeviceToHost);
	printf("cudaError %s\n", cudaGetErrorString(cudaGetLastError()));
	//cudaMalloc(&cumemtest, 7066681344ll);
	printf("cudaError %s %d\n", cudaGetErrorString(cudaGetLastError()), sizeof(Bridge));
	getc(stdin);
	int n = 1922580;
	int64_t *d_mpidx;
	int** d_mp, *d_cmb, *d_cof, *d_fin, *d_idxs;
	const int* _true = new int(1);
	int h_fin = false, h_cmb = 0;
	void* h_dmemaddress = 0;
	curandStateMRG32k3a_t *state;
	const int poolsize = 2097152, max_page = 64;
	cudaMalloc(&d_mpidx, sizeof(int64_t) * n);
	cudaMalloc(&d_mp, sizeof(int*) * max_page);
	cudaMalloc(&d_cmb, sizeof(int));
	cudaMalloc(&d_cof, sizeof(int));
	cudaMalloc(&d_fin, sizeof(int));
	cudaMalloc(&state, sizeof(curandStateMRG32k3a_t));
	cudaMalloc(&d_idxs, sizeof(int) * 65536);

	cudaMalloc(&h_dmemaddress, 8388608);
	cudaMemset(d_idxs, 0, 65536 * sizeof(int));
	cudaMemcpy(d_mp + h_cmb, &h_dmemaddress, sizeof(int*), cudaMemcpyHostToDevice);
	h_cmb++;
	memorypool_init << <128, 128 >> > (d_mp, d_cmb, d_cof, state);
	cudaDeviceSynchronize();
	printf("cudaError %s %d\n", cudaGetErrorString(cudaGetLastError()), sizeof(Bridge));

	cudaError_t lastError = cudaSuccess;
	while (!h_fin && lastError == cudaSuccess) {
		
		cudaMemset(d_fin, 1, 4);
		cudaMalloc(&h_dmemaddress, 8388608);
		cudaMemcpy(d_mp + h_cmb, &h_dmemaddress, sizeof(int*), cudaMemcpyHostToDevice);
		h_cmb++;

		parallelized_memory_allocation_test<<<128, 128>>>
			(d_mp, d_mpidx, d_cmb, d_cof, n, d_idxs, poolsize, state, d_fin);
		cudaMemcpy(&h_fin, d_fin, sizeof(int), cudaMemcpyDeviceToHost);
		substract << <1, 1 >> >(d_cmb, d_cof, poolsize);
		lastError = cudaGetLastError();
		printf("cudaError %s\n", cudaGetErrorName(lastError));

	}
	printf("cudaError %s\n", cudaGetErrorName(cudaGetLastError()));

}