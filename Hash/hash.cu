/*
 * hash.cu
 *
 *  Created on: 2018-9-4
 *      Author: qiushuang
 */

/*
 * This hash table construction method is based on the gcc libiberty hash.
 * Hash function used in searching is murmur hashing, favoring its value distribution and collision properties.
 */

#include <cuda_runtime.h>
#include "hash.h"

#define DEBUG

#ifndef HASH_CUH
#define HASH_CUH

#define THREADS_PER_BLOCK 1024
#define MAX_NUM_BLOCKS 64 // maximum number of blocks
#define MAX_NUM_THREADS (MAX_NUM_BLOCKS*THREADS_PER_BLOCK)
#define NUM_WARPS_PER_BLOCK 32
#define WARP_WIDTH 32
#define FULL_MASK 0xffffffff

#define SLOT_WIDTH 32 // number of slots per warp

typedef uint key_type;
typedef uint value_type;
typedef int state_type;
typedef uint slot_mask_type;

typedef struct entry
{
	state_type state_flag; // 0: empty; 1: locked; 2: occupied
	key_type key;
	value_type value;
} entry_t;

typedef struct entry_soa
{
	int * state_flags;
	key_type * keys;
	value_type * values;
} entry_soa_t;

typedef struct entry_kv
{
	key_type key;
	value_type value;
} entry_kv_t;

enum class device_type {cpu, gpu};

__constant__ static entry_t * dtab; // hash table pointer on the GPU
__constant__ static size_t dsize; // hash table size on the GPU
__constant__ static prime_index_t d_size_prime_index; // size prime index on the GPU
//* if using structure of arrays hash table: *
__constant__ static int * state_flags;
__constant__ static key_type * keys;
__constant__ static value_type * values;
//* using warp-centric hash table probing
__constant__ static entry_kv_t * htable;

static const int HMultiplyDeBruijnBitPosition[32] =
{
  0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
  31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
};

/* Prime table on the host */
static const hashval_t hprime_tab[30] = {
            7,
           13,
           31,
           61,
          127,
          251,
          509,
         1021,
         2039,
         4093,
         8191,
        16381,
        32749,
        65521,
       131071,
       262139,
       524287,
      1048573,
      2097143,
      4194301,
      8388593,
     16777213,
     33554393,
     67108859,
    134217689,
    268435399,
    536870909,
   1073741789,
   2147483647,
  /* Avoid "decimal constant so large it is unsigned" for 4294967291.  */
   0xfffffffb
};

__constant__ static const int MultiplyDeBruijnBitPosition[32] =
{
  0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
  31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
};

/* Prime table on the GPU */
__constant__ static const hashval_t prime_tab[30] = {
            7,
           13,
           31,
           61,
          127,
          251,
          509,
         1021,
         2039,
         4093,
         8191,
        16381,
        32749,
        65521,
       131071,
       262139,
       524287,
      1048573,
      2097143,
      4194301,
      8388593,
     16777213,
     33554393,
     67108859,
    134217689,
    268435399,
    536870909,
   1073741789,
   2147483647,
  /* Avoid "decimal constant so large it is unsigned" for 4294967291.  */
   0xfffffffb
};

/* The following function returns an index into the above table of the
   nearest prime number which is greater than N, and near a power of two. */
__host__ static uint
higher_prime_index (unsigned long n)
{
  unsigned int low = 0;
  unsigned int high = sizeof(hprime_tab) / sizeof(hashval_t);

  while (low != high)
    {
      unsigned int mid = low + (high - low) / 2;
      if (n > hprime_tab[mid])
    	  low = mid + 1;
      else
    	  high = mid;
    }

  /* If we've run out of primes, abort.  */
  if (n > hprime_tab[low])
    {
      printf ("Cannot find prime bigger than %lu\n", n);
    }

  return low;
}

class hashtab_t
{
private:
	entry_t * tab; // hash table pointer, either point to device or point to a host hash table
	size_t size; // hash table size
	prime_index_t size_prime_index; // size prime index on the host
	device_type device; // CPU or GPU

public:
	__host__ void createHashtab (size_t num_of_elems, device_type d)
	{
		uint h_size_prime_index;
		hashsize_t table_size; //primary table size
		num_of_elems = (num_of_elems * ELEM_FACTOR + SLOT_WIDTH - 1) / SLOT_WIDTH;
		h_size_prime_index = higher_prime_index (num_of_elems * ELEM_FACTOR);
		table_size = hprime_tab[h_size_prime_index];

		if (d == device_type::gpu)
		{
			CUDA_CHECK_RETURN (cudaMalloc (&tab, sizeof(entry_t) * SLOT_WIDTH * table_size));
			CUDA_CHECK_RETURN (cudaMemset (tab, 0, sizeof(entry_t) * SLOT_WIDTH * table_size));
			CUDA_CHECK_RETURN (cudaMemcpyToSymbol (dtab, &tab, sizeof(entry_t *)));
			CUDA_CHECK_RETURN (cudaMemcpyToSymbol (d_size_prime_index, &h_size_prime_index, sizeof(prime_index_t)));
			CUDA_CHECK_RETURN (cudaMemcpyToSymbol (dsize, &table_size, sizeof(size_t)));
		}
		else
		{
			tab = (entry_t *) malloc (sizeof(entry_t) * SLOT_WIDTH * table_size);
			CHECK_PTR_RETURN (tab, "malloc hash table on the host error!\n");
		}
		size = table_size;
		size_prime_index = h_size_prime_index;
		device = d;
	}

	__host__ void destoryHashtab (void)
	{
		if (device == device_type::gpu)
		{
			cudaFree (tab);
		}
		else
		{
			free (tab);
		}
	}

	__host__ entry_t * getDeviceTab (void) { return tab; }
	__host__ size_t getTabSize (void) { return size; }
};

class hashtab_soa_t
{
private:
	entry_soa_t tab; // structure of arrays of hash table, accessible only on the host
	size_t size; // hash table size
	prime_index_t size_prime_index; // size prime index on the host
	device_type device; // CPU or GPU

public:
	__host__ void createHashtab (size_t num_of_elems, device_type d)
	{
		uint h_size_prime_index;
		hashsize_t table_size; //primary table size
		num_of_elems = (num_of_elems + SLOT_WIDTH - 1) / SLOT_WIDTH;
		h_size_prime_index = higher_prime_index (num_of_elems * ELEM_FACTOR);
		table_size = hprime_tab[h_size_prime_index];

		if (d == device_type::gpu)
		{
			CUDA_CHECK_RETURN (cudaMalloc (&(tab.state_flags), sizeof(state_type) * SLOT_WIDTH * table_size));
			CUDA_CHECK_RETURN (cudaMemset (tab.state_flags, 0, sizeof(state_type) * SLOT_WIDTH * table_size));
			CUDA_CHECK_RETURN (cudaMalloc (&(tab.keys), sizeof(key_type) * SLOT_WIDTH * table_size));
			CUDA_CHECK_RETURN (cudaMalloc (&(tab.values), sizeof(value_type) * SLOT_WIDTH * table_size));
			CUDA_CHECK_RETURN (cudaMemcpyToSymbol (state_flags, &(tab.state_flags), sizeof(state_type*)));
			CUDA_CHECK_RETURN (cudaMemcpyToSymbol (keys, &(tab.keys), sizeof(key_type*)));
			CUDA_CHECK_RETURN (cudaMemcpyToSymbol (values, &(tab.values), sizeof(value_type*)));

			CUDA_CHECK_RETURN (cudaMemcpyToSymbol (d_size_prime_index, &h_size_prime_index, sizeof(prime_index_t)));
			CUDA_CHECK_RETURN (cudaMemcpyToSymbol (dsize, &table_size, sizeof(size_t)));
		}
		else
		{
			tab.state_flags = (int *) malloc (sizeof(state_type) * SLOT_WIDTH * table_size);
			CHECK_PTR_RETURN (tab.state_flags, "malloc hash table state_flags on the host error!\n");
			tab.keys = (key_type *) malloc (sizeof(key_type) * SLOT_WIDTH * table_size);
			CHECK_PTR_RETURN (tab.keys, "malloc hash table keys on the host error!\n");
			tab.values = (value_type *) malloc (sizeof(value_type) * SLOT_WIDTH * table_size);
			CHECK_PTR_RETURN (tab.values, "malloc hash table values on the host error!\n");

		}
		size = table_size;
		size_prime_index = h_size_prime_index;
		device = d;
	}

	__host__ void destoryHashtab (void)
	{
		if (device == device_type::gpu)
		{
			cudaFree (tab.values);
			cudaFree (tab.keys);
			cudaFree (tab.state_flags);
		}
		else
		{
			free (tab.values);
			free (tab.keys);
			free (tab.state_flags);
		}
	}

	__host__ key_type * getDeviceKeys (void) { return tab.keys; }
	__host__ value_type * getDeviceValues (void) { return tab.values; }
	__host__ int * getDeviceStates (void) { return tab.state_flags; }
	__host__ size_t getTabSize (void) { return size; }
};

class hashtab_kv_t
{
private:
	entry_kv_t * tab; // hash table pointer, either point to device or point to a host hash table
	size_t size; // hash table size
	prime_index_t size_prime_index; // size prime index on the host
	device_type device; // CPU or GPU

public:
	__host__ void createHashtab (size_t num_of_elems, device_type d)
	{
		uint h_size_prime_index;
		hashsize_t table_size; //primary table size
		num_of_elems = (num_of_elems * ELEM_FACTOR + SLOT_WIDTH - 1) / SLOT_WIDTH;
		h_size_prime_index = higher_prime_index (num_of_elems * ELEM_FACTOR);
		table_size = hprime_tab[h_size_prime_index];

		if (d == device_type::gpu)
		{
			CUDA_CHECK_RETURN (cudaMalloc (&tab, (sizeof(entry_kv_t) * SLOT_WIDTH + sizeof(slot_mask_type) + sizeof(state_type)) * table_size));
			CUDA_CHECK_RETURN (cudaMemset (tab, 0, (sizeof(entry_kv_t) * SLOT_WIDTH + sizeof(slot_mask_type) + sizeof(state_type)) * table_size));
			CUDA_CHECK_RETURN (cudaMemcpyToSymbol (htable, &tab, sizeof(entry_kv_t *)));
			CUDA_CHECK_RETURN (cudaMemcpyToSymbol (d_size_prime_index, &h_size_prime_index, sizeof(prime_index_t)));
			CUDA_CHECK_RETURN (cudaMemcpyToSymbol (dsize, &table_size, sizeof(size_t)));
		}
		else
		{
			tab = (entry_kv_t *) malloc ((sizeof(entry_kv_t) * SLOT_WIDTH + sizeof(state_type) + sizeof(slot_mask_type)) * table_size);
			CHECK_PTR_RETURN (tab, "malloc hash table on the host error!\n");
		}
		size = table_size;
		size_prime_index = h_size_prime_index;
		device = d;
	}

	__host__ void destoryHashtab (void)
	{
		if (device == device_type::gpu)
		{
			cudaFree (tab);
		}
		else
		{
			free (tab);
		}
	}

	__host__ entry_kv_t * getDeviceTab (void) { return tab; }
	__host__ size_t getTabSize (void) { return size; }
};


//*??? More work on this function! *
__device__ __host__ static inline hashval_t
hashtab_mod_1 (hashval_t x, hashval_t y)
{
  /* The multiplicative inverses computed above are for 32-bit types, and
     requires that we be able to compute a highpart multiply.  */
  /* Otherwise just use the native division routines.  */
  return x % y;
}

//* Compute the primary hash for HASH given HASHTAB's current size.  *
__device__ static inline hashval_t
hashtab_mod (hashval_t hash, uint size_prime_index)
{
  return hashtab_mod_1 (hash, prime_tab[size_prime_index]);
}

//* Compute the secondary hash for HASH given HASHTAB's current size.  *
__device__ static inline hashval_t
hashtab_mod_m2 (hashval_t hash, uint size_prime_index)
{
  return 1 + hashtab_mod_1 (hash, prime_tab[size_prime_index] - 2);
}

template <typename T>
__device__ inline T insert_int_key (volatile T * key, T new_key)
{
	if (*key != 0)
		printf("error!!!!!!!!!!\n");
	*key = new_key; // volatile T
	return (*key);
//	T ret = atomicExch (key, new_key);
//	if (*key == 0 && new_key != 0)
//		printf ("Inserting new key failed!\n");
//	return ret;

}

__device__ inline void update_slot_mask (volatile slot_mask_type * mask, int pos)
{
	(*mask) |= (1 << (sizeof(*mask) * BYTE_BIT - pos - 1));
}

template <typename T>
__device__ __host__ inline int compare_keys (T * key1, T * key2)
{
	if (*key1 == *key2)
		return 1;
	else return 0;
}

template <typename T>
__device__ inline T replace_int_value (T * value, T new_value)
{
	T ret = atomicExch (value, new_value);
	return ret;
}


//* Replace-value based hash table with one thread *
__device__ bool insert_key_value (key_type * key, value_type * value)
{
	hashval_t hash = murmur_hash3_32 (key, DEFAULT_SEED);
	hashval_t index1, index2;
	entry_t * entry;
	entry_t * entries = dtab;
	prime_index_t size_prime_index = d_size_prime_index;
	hashval_t table_size = dsize;

	int flag = -1;
	index1 = hashtab_mod (hash, size_prime_index); // primary index
	/* debug only: */
	if (index1 >= table_size)
	{
		return false;
	}
	index2 = fnv_hash_32((char *)key) % SLOT_WIDTH; // secondary index

	//* first try of insertion: *
	entry = entries + index1*SLOT_WIDTH + index2;
	flag = atomicCAS(&(entry->state_flag), 0, 1);
	if (flag == 0) // empty
	{
		insert_int_key<key_type> (&(entry->key), *key);
		int ret = atomicCAS (&(entry->state_flag), 1, 2);
		if (ret != 1)
		{
			printf ("Error in changing state_flag from locked to occupied!\n");
		}
	}
	while ((flag = atomicCAS(&(entry->state_flag), 2, 2)) != 2) {}

	if (compare_keys<key_type>(key, &(entry->key)))
	{
		replace_int_value<key_type> (&(entry->value), *value);
		return true;
	}
	//* otherwise, probe secondary hash slots *
	for (int s=0; s<SLOT_WIDTH; s++)
	{
		if (s==index2) continue;
		entry = entries + index1*SLOT_WIDTH + s;
		flag = atomicCAS(&(entry->state_flag), 0, 1);
		if (flag == 0) // empty
		{
			insert_int_key<key_type> (&(entry->key), *key);
			int ret = atomicCAS (&(entry->state_flag), 1, 2);
			if (ret != 1)
			{
				printf ("Error in changing state_flag from locked to occupied!\n");
			}
		}
		while ((flag = atomicCAS(&(entry->state_flag), 2, 2)) != 2) {}
		if (compare_keys<key_type>(key, &(entry->key)))
		{
			replace_int_value<key_type> (&(entry->value), *value);
			return true;
		}
	}

	//* first try fails, continue with secondary hash2 *
	hashval_t hash2 = hashtab_mod_m2 (hash, size_prime_index);
	for (size_t i=0; i<table_size; i++)
	{
		index1 += hash2;
		if (index1 >= table_size)
			index1 -= table_size;

		entry_t * entry = entries + index1*SLOT_WIDTH + index2;
		flag = atomicCAS(&(entry->state_flag), 0, 1);
		if (flag == 0) // empty
		{
			insert_int_key<key_type> (&(entry->key), *key);
			int ret = atomicCAS (&(entry->state_flag), 1, 2);
			if (ret != 1)
			{
				printf ("Error in changing state_flag from locked to occupied!\n");
			}
		}
		while ((flag = atomicCAS(&(entry->state_flag), 2, 2)) != 2) {}
		if (compare_keys<key_type>(key, &(entry->key)))
		{
			replace_int_value<key_type> (&(entry->value), *value);
			return true;
		}
		//* otherwise, probe secondary hash slots *
		for (int s=0; s<SLOT_WIDTH; s++)
		{
			if (s==index2) continue;
			entry = entries + index1*SLOT_WIDTH + s;
			flag = atomicCAS(&(entry->state_flag), 0, 1);
			if (flag == 0) // empty
			{
				insert_int_key<key_type> (&(entry->key), *key);
				int ret = atomicCAS (&(entry->state_flag), 1, 2);
				if (ret != 1)
				{
					printf ("Error in changing state_flag from locked to occupied!\n");
				}
			}
			while ((flag = atomicCAS(&(entry->state_flag), 2, 2)) != 2) {}
			if (compare_keys<key_type>(key, &(entry->key)))
			{
				replace_int_value<key_type> (&(entry->value), *value);
				return true;
			}
		}
	}
	return false;
}

//* Replace-value based hash table with one thread *
__device__ bool insert_key_value_soa (key_type * key, value_type * value)
{
	hashval_t hash = murmur_hash3_32 (key, DEFAULT_SEED);
	hashval_t index1, index2;

	key_type * dtab_key = keys;
	value_type * dtab_value = values;
	int * dtab_state = state_flags;
	prime_index_t size_prime_index = d_size_prime_index;
	hashval_t table_size = dsize;
	key_type * key_slot;
	value_type * value_slot;
	int * state_slot;

	int flag = -1;
	index1 = hashtab_mod (hash, size_prime_index); // primary index
	/* debug only: */
	if (index1 >= table_size)
	{
		return false;
	}
	index2 = fnv_hash_32((char *)key) % SLOT_WIDTH; // secondary index

	//* first try of insertion: *
	key_slot = dtab_key + index1 * SLOT_WIDTH + index2;
	value_slot = dtab_value + index1 * SLOT_WIDTH + index2;
	state_slot = dtab_state + index1 * SLOT_WIDTH + index2;
	flag = atomicCAS(state_slot, 0, 1);
	if (flag == 0) // empty
	{
		insert_int_key<key_type> (key_slot, *key);
		int ret = atomicCAS (state_slot, 1, 2);
		if (ret != 1)
		{
			printf ("Error in changing state_flag from locked to occupied!\n");
		}
	}
	while ((flag = atomicCAS(state_slot, 2, 2)) != 2) {}

	if (compare_keys<key_type>(key, key_slot))
	{
		replace_int_value<key_type> (value_slot, *value);
		return true;
	}
	//* otherwise, probe secondary hash slots *
	for (int s=0; s<SLOT_WIDTH; s++)
	{
		if (s==index2) continue;
		key_slot = dtab_key + index1 * SLOT_WIDTH + s;
		value_slot = dtab_value + index1 * SLOT_WIDTH + s;
		state_slot = dtab_state + index1 * SLOT_WIDTH + s;
		flag = atomicCAS(state_slot, 0, 1);
		if (flag == 0) // empty
		{
			insert_int_key<key_type> (key_slot, *key);
			int ret = atomicCAS (state_slot, 1, 2);
			if (ret != 1)
			{
				printf ("Error in changing state_flag from locked to occupied!\n");
			}
		}
		while ((flag = atomicCAS(state_slot, 2, 2)) != 2) {}
		if (compare_keys<key_type>(key, key_slot))
		{
			replace_int_value<key_type> (value_slot, *value);
			return true;
		}
	}

	//* first try fails, continue with secondary hash2 *
	hashval_t hash2 = hashtab_mod_m2 (hash, size_prime_index);
	for (size_t i=0; i<table_size; i++)
	{
		index1 += hash2;
		if (index1 >= table_size)
			index1 -= table_size;

		key_slot = dtab_key + index1 * SLOT_WIDTH + index2;
		value_slot = dtab_value + index1 * SLOT_WIDTH + index2;
		state_slot = dtab_state + index1 * SLOT_WIDTH + index2;
		flag = atomicCAS(state_slot, 0, 1);
		if (flag == 0) // empty
		{
			insert_int_key<key_type> (key_slot, *key);
			int ret = atomicCAS (state_slot, 1, 2);
			if (ret != 1)
			{
				printf ("Error in changing state_flag from locked to occupied!\n");
			}
		}
		while ((flag = atomicCAS(state_slot, 2, 2)) != 2) {}
		if (compare_keys<key_type>(key, key_slot))
		{
			replace_int_value<key_type> (value_slot, *value);
			return true;
		}
		//* otherwise, probe secondary hash slots *
		for (int s=0; s<SLOT_WIDTH; s++)
		{
			if (s==index2) continue;
			key_slot = dtab_key + index1 * SLOT_WIDTH + s;
			value_slot = dtab_value + index1 * SLOT_WIDTH + s;
			state_slot = dtab_state + index1 * SLOT_WIDTH + s;
			flag = atomicCAS(state_slot, 0, 1);
			if (flag == 0) // empty
			{
				insert_int_key<key_type> (key_slot, *key);
				int ret = atomicCAS (state_slot, 1, 2);
				if (ret != 1)
				{
					printf ("Error in changing state_flag from locked to occupied!\n");
				}
			}
			while ((flag = atomicCAS(state_slot, 2, 2)) != 2) {}
			if (compare_keys<key_type>(key, key_slot))
			{
				replace_int_value<key_type> (value_slot, *value);
				return true;
			}
		}
	}
	return false;
}


__global__ void insert_key_value_hashtab (key_type * keys, value_type * values, size_t num_of_keys)
{
	uint gid = blockDim.x * blockIdx.x + threadIdx.x;

	size_t keys_per_thread = (num_of_keys + MAX_NUM_THREADS - 1) / MAX_NUM_THREADS; // keys per thread
	if (gid==0)
	{
		printf ("number of keys per thread: %u\n", keys_per_thread);
	}
	for (size_t i = 0; i < keys_per_thread; i++)
	{
		if (gid * keys_per_thread + i >= num_of_keys)
			return;

		if (insert_key_value (&keys[gid*keys_per_thread + i], &values[gid*keys_per_thread+i]) == false)
		{
			printf ("CAREFUL!!!!!!! INSERTION FAILS!\n");
		}
	}
}

__global__ void insert_key_value_hashtab_soa (key_type * keys, value_type * values, size_t num_of_keys)
{
	uint gid = blockDim.x * blockIdx.x + threadIdx.x;

	size_t keys_per_thread = (num_of_keys + MAX_NUM_THREADS - 1) / MAX_NUM_THREADS; // keys per thread
	if (gid==0)
	{
		printf ("number of keys per thread: %u\n", keys_per_thread);
	}
	for (size_t i = 0; i < keys_per_thread; i++)
	{
		if (gid * keys_per_thread + i >= num_of_keys)
			return;

		if (insert_key_value_soa (&keys[gid*keys_per_thread + i], &values[gid*keys_per_thread+i]) == false)
		{
			printf ("CAREFUL!!!!!!! INSERTION FAILS!\n");
		}
	}

}

__global__ void insert_key_value_hashtab_warp (key_type * input_keys, value_type * input_values, size_t num_of_keys)
{
	uint gid = blockDim.x * blockIdx.x + threadIdx.x;
	uint tid = threadIdx.x;
	uint warpid = gid / WARP_WIDTH;
	uint laneid = tid % WARP_WIDTH;

	size_t keys_per_warp = (num_of_keys + (MAX_NUM_THREADS/WARP_WIDTH - 1)) / (MAX_NUM_THREADS/WARP_WIDTH); // keys per block
//	keys_per_warp = (keys_per_warp + WARP_WIDTH - 1) / WARP_WIDTH;
	uint j;
	for (j = 0; j < keys_per_warp; j++)
	{
		if (warpid * keys_per_warp + j >= num_of_keys)
			return;

		key_type * key = &(input_keys[warpid * keys_per_warp + j]);
		value_type * value = &(input_values[warpid * keys_per_warp + j]);

		hashval_t hash = murmur_hash3_32 (key, DEFAULT_SEED);
		hashval_t index1;
		void * ptr;
		state_type * state_ptr;
		slot_mask_type * slot_mask_ptr;
		entry_kv_t * entries = htable;
		entry_kv_t * entry;
		prime_index_t size_prime_index = d_size_prime_index;
		hashval_t table_size = dsize;
		int cont_flag = 0; // continue flag to make warp continue synchronously in a loop
		bool next_flag = false;
		uint active;
		state_type flag = -1;

		index1 = hashtab_mod (hash, size_prime_index); // primary index
		/* debug only: */
		if (index1 >= table_size)
		{
			continue;
		}

		//* first try of insertion: *
		ptr = (void*)((char*)entries + index1 * (sizeof(entry_kv_t) * SLOT_WIDTH + sizeof(slot_mask_type) + sizeof(state_type)));
		state_ptr = (state_type *)ptr;
//		slot_mask_ptr = (slot_mask_type*)(state_ptr + 1);
		slot_mask_ptr = (slot_mask_type*)((char*)state_ptr + sizeof(state_type));
//		entry = (entry_kv_t*)(slot_mask_ptr + 1) + laneid;
		entry = (entry_kv_t*)((char*)slot_mask_ptr + sizeof(slot_mask_type) + sizeof(entry_kv_t) * laneid);

//		while (atomicCAS(slot_mask_ptr, FULL_MASK, FULL_MASK) != FULL_MASK)
		while (*slot_mask_ptr != FULL_MASK)
		{
			if (laneid == 0)
			{
				flag = atomicCAS(state_ptr, 0, 1);
			}
			flag = __shfl_sync (FULL_MASK, flag, 0, WARP_WIDTH); // broadcast flag to threads in this warp
//			active = __shfl_sync (FULL_MASK, *slot_mask_ptr, 0, WARP_WIDTH);
			active = *slot_mask_ptr;
			if (flag == 0) // empty slot exists
			{
				uint r = MultiplyDeBruijnBitPosition[((uint)(((~active) & -(~active)) * 0x077CB531U)) >> 27];
				if ((~active) && r == WARP_WIDTH - 1 - laneid) // thread selected for insertion
				{
					insert_int_key<key_type> (&(entry->key), *key);
					update_slot_mask (slot_mask_ptr, laneid);
					int ret = atomicCAS (state_ptr, 1, 0);
					if (ret != 1)
					{
						printf ("Error in changing state_flag from locked to occupied!\n");
					}
					replace_int_value<value_type> (&(entry->value), *value);
					cont_flag = 1;
				}
				cont_flag = __shfl_sync (FULL_MASK, cont_flag, WARP_WIDTH-1-r, WARP_WIDTH);
				if (cont_flag)
				{
					next_flag = true;
					break;
				}
			}
			if (laneid == 0)
			{
				while ((flag = atomicCAS(state_ptr, 0, 0)) != 0) {}
			}
			flag = __shfl_sync (FULL_MASK, flag, 0, WARP_WIDTH); // broadcast flag to threads in this warp
//			active = __shfl_sync (FULL_MASK, *slot_mask_ptr, 0, WARP_WIDTH);
//			active = *slot_mask_ptr;
			int ret = 0;
			ret = compare_keys (key, &(entry->key));
/*			if (active & (1 << (WARP_WIDTH - 1 - laneid)))
			{
				ret = compare_keys (key, &(entry->key));
				if (ret==1)
					printf ("entry->key = %u, key = %u\n", entry->key, *key);
			}*/
			uint ballot_result = __ballot_sync (FULL_MASK, ret);
			uint r = MultiplyDeBruijnBitPosition[((uint)((ballot_result & -ballot_result) * 0x077CB531U)) >> 27];
			if (ballot_result && r == WARP_WIDTH - 1 - laneid)
			{
				replace_int_value<value_type> (&(entry->value), *value);
				cont_flag = 1;
			}
			cont_flag = __shfl_sync (FULL_MASK, cont_flag, WARP_WIDTH-1-r, WARP_WIDTH);
			if (cont_flag)
			{
				next_flag = true;
				break;
			}
		}
		if (next_flag)
			continue;

		//* first try fails, continue with secondary hash2 *
		hashval_t hash2 = hashtab_mod_m2 (hash, size_prime_index);
		size_t i;
		for (i=0; i<table_size; i++)
		{
			index1 += hash2;
			if (index1 >= table_size)
				index1 -= table_size;

			ptr = (void*)((char*)entries + index1 * (sizeof(entry_kv_t) * SLOT_WIDTH + sizeof(slot_mask_type) + sizeof(state_type)));
			state_ptr = (state_type *)ptr;
			slot_mask_ptr = (slot_mask_type*)((char*)state_ptr + sizeof(state_type));
			entry = (entry_kv_t*)((char*)slot_mask_ptr + sizeof(slot_mask_type) + sizeof(entry_kv_t) * laneid);
//			while (atomicCAS(slot_mask_ptr, FULL_MASK, FULL_MASK) != FULL_MASK)
			while (*slot_mask_ptr != FULL_MASK)
			{
				if (laneid == 0)
				{
					flag = atomicCAS(state_ptr, 0, 1);
				}
				flag = __shfl_sync (FULL_MASK, flag, 0, WARP_WIDTH); // broadcast flag to threads in this warp
//				active = __shfl_sync (FULL_MASK, *slot_mask_ptr, 0, WARP_WIDTH);
				active = *slot_mask_ptr;
				if (flag == 0) // empty slot exists
				{
					uint r = MultiplyDeBruijnBitPosition[((uint)(((~active) & -(~active)) * 0x077CB531U)) >> 27];
					if ((~active) && r == WARP_WIDTH - 1 - laneid) // thread selected for insertion
					{
						insert_int_key<key_type> (&(entry->key), *key);
						update_slot_mask (slot_mask_ptr, laneid);
						int ret = atomicCAS (state_ptr, 1, 0);
						if (ret != 1)
						{
							printf ("Error in changing state_flag from locked to occupied!\n");
						}
						replace_int_value<value_type> (&(entry->value), *value);
						cont_flag = 1;
					}
					cont_flag = __shfl_sync (FULL_MASK, cont_flag, WARP_WIDTH-1-r, WARP_WIDTH);
					if (cont_flag)
					{
						next_flag = true;
						break;
					}
				}
				if (laneid == 0)
				{
					while ((flag = atomicCAS(state_ptr, 0, 0)) != 0) {}
				}
				flag = __shfl_sync (FULL_MASK, flag, 0, WARP_WIDTH); // broadcast flag to threads in this warp
//				active = __shfl_sync (FULL_MASK, *slot_mask_ptr, 0, WARP_WIDTH);
//				active = *slot_mask_ptr;
				int ret = 0;
				ret = compare_keys (key, &(entry->key));
/*				if (active & (1 << (WARP_WIDTH - 1 - laneid)))
				{
					ret = compare_keys (key, &(entry->key));
					if (ret==1)
						printf ("entry->key = %u, key = %u\n", entry->key, *key);
				}*/
				uint ballot_result = __ballot_sync (FULL_MASK, ret);
				uint r = MultiplyDeBruijnBitPosition[((uint)((ballot_result & -ballot_result) * 0x077CB531U)) >> 27];
				if (ballot_result && r == WARP_WIDTH - 1 - laneid)
				{
					replace_int_value<value_type> (&(entry->value), *value);
					cont_flag = 1;
				}
				cont_flag = __shfl_sync (FULL_MASK, cont_flag, WARP_WIDTH-1-r, WARP_WIDTH);
				if (cont_flag)
				{
					next_flag = true;
					break;
				}
			}
			if (next_flag)
				break;
		}
		if (i==table_size && next_flag==false)
			printf ("error! space not found!!!\n");
	}
}

extern "C"
{
#ifdef DEBUG
__host__ void check_hash_results (entry_t * tab, size_t size)
{
	printf ("HASH TABLE SIZE:::::::: %lu\n", size);
	size_t count=0;
	FILE * tabfile;
	if ((tabfile = fopen("/home/sqiuac/common/htab", "w")) == NULL)
	{
		printf ("ERROR!!! Can not open hash table file for checking results!\n");
		exit(0);
	}
	for (size_t i=0; i<size; i++)
	{
		if (tab[i].state_flag == 2)
		{
			count++;
//			fprintf(tabfile, "%u, %u\n", tab[i].key, tab[i].value);
		}
	}
	printf ("NUMBER OF ELEMENTS IN HASH TABLE: %lu\n", count);
	fclose(tabfile);
}

__host__ void check_hash_results_soa (int * states, key_type * keys, value_type * values, size_t size)
{
	printf ("HASH TABLE SIZE:::::::: %lu\n", size);
	size_t count=0;
	FILE * tabfile;
	if ((tabfile = fopen("/home/sqiuac/common/htab", "w")) == NULL)
	{
		printf ("ERROR!!! Can not open hash table file for checking results!\n");
		exit(0);
	}
	for (size_t i=0; i<size; i++)
	{
		if (states[i] == 2)
		{
			count++;
			fprintf(tabfile, "%u, %u\n", keys[i], values[i]);
		}
	}
	printf ("NUMBER OF ELEMENTS IN HASH TABLE: %lu\n", count);
	fclose(tabfile);

	fopen("/home/sqiuac/common/stand", "w");
	for (size_t i=0; i<size; i++)
	{
		fprintf (tabfile, "%u, %u\n", i+1, i+2);
	}
	fclose(tabfile);
}

__host__ void check_hash_results_warp (entry_kv_t * tab, size_t size)
{
	printf ("HASH TABLE SIZE:::::::: %lu\n", size);
	size_t count=0;
	FILE * tabfile;
	void * ptr = tab;
	if ((tabfile = fopen("/home/sqiuac/common/htab", "w")) == NULL)
	{
		printf ("ERROR!!! Can not open hash table file for checking results!\n");
		exit(0);
	}
	for (size_t i=0; i<size; i++)
	{
//		if (*(state_type*)ptr == 2)
		{
			ptr = ptr + sizeof(state_type);
			slot_mask_type slot_mask = *(slot_mask_type*)ptr;
			ptr = ptr + sizeof(slot_mask_type);
			for (int j=0; j<SLOT_WIDTH; j++)
			{
				if(slot_mask & (1 << (sizeof(slot_mask)*BYTE_BIT - j - 1)))
				{
					count++;
//					fprintf (tabfile, "%u, %u\n", ((entry_kv_t*)ptr)->key, ((entry_kv_t*)ptr)->value);
				}
				ptr = ptr + sizeof(entry_kv_t);
			}
		}
	}
	printf ("NUMBER OF ELEMENTS IN HASH TABLE: %lu\n", count);
	fclose(tabfile);
}

#endif

__host__ void construct_hashtab_gpu (key_type * keys, value_type * values, size_t num_of_keys)
{
	hashtab_t htab;
	key_type * dkeys = NULL;
	value_type * dvalues = NULL;
	CUDA_CHECK_RETURN (cudaMalloc (&dkeys, sizeof(key_type) * num_of_keys));
	CUDA_CHECK_RETURN (cudaMemcpy (dkeys, keys, sizeof(key_type) * num_of_keys, cudaMemcpyHostToDevice));
	if(values != NULL)
	{
		CUDA_CHECK_RETURN (cudaMalloc (&dvalues, sizeof(value_type) * num_of_keys));
		CUDA_CHECK_RETURN (cudaMemcpy (dvalues, values, sizeof(value_type) * num_of_keys, cudaMemcpyHostToDevice));
	}
	htab.createHashtab(num_of_keys, device_type::gpu);
	insert_key_value_hashtab<<<MAX_NUM_BLOCKS, THREADS_PER_BLOCK>>>(dkeys, dvalues, num_of_keys);

#ifdef DEBUG
	size_t size = htab.getTabSize();
	printf ("primary table size::::: %lu\n", size);
	size = size * SLOT_WIDTH;
	entry_t * table = (entry_t *) malloc (sizeof(entry_t) * size);
	CUDA_CHECK_RETURN (cudaMemcpy(table, htab.getDeviceTab(), sizeof(entry_t) * size, cudaMemcpyDeviceToHost));
	check_hash_results(table, size);
	free (table);
#endif

	htab.destoryHashtab();
}

__host__ void construct_hashtab_warp_gpu (key_type * keys, value_type * values, size_t num_of_keys)
{
	hashtab_kv_t htab;
	key_type * dkeys = NULL;
	value_type * dvalues = NULL;
	CUDA_CHECK_RETURN (cudaMalloc (&dkeys, sizeof(key_type) * num_of_keys));
	CUDA_CHECK_RETURN (cudaMemcpy (dkeys, keys, sizeof(key_type) * num_of_keys, cudaMemcpyHostToDevice));
	if(values != NULL)
	{
		CUDA_CHECK_RETURN (cudaMalloc (&dvalues, sizeof(value_type) * num_of_keys));
		CUDA_CHECK_RETURN (cudaMemcpy (dvalues, values, sizeof(value_type) * num_of_keys, cudaMemcpyHostToDevice));
	}
	htab.createHashtab(num_of_keys, device_type::gpu);
	insert_key_value_hashtab_warp<<<MAX_NUM_BLOCKS, THREADS_PER_BLOCK>>>(dkeys, dvalues, num_of_keys);

#ifdef DEBUG
	size_t size = htab.getTabSize();
	printf ("primary table size::::: %lu\n", size);
	entry_kv_t * table = (entry_kv_t *) malloc ((sizeof(entry_kv_t) * SLOT_WIDTH + sizeof(slot_mask_type) + sizeof(state_type)) * size);
	CUDA_CHECK_RETURN (cudaMemcpy(table, htab.getDeviceTab(), (sizeof(entry_kv_t) * SLOT_WIDTH + sizeof(slot_mask_type) + sizeof(state_type)) * size, cudaMemcpyDeviceToHost));
	check_hash_results_warp(table, size);
	free (table);
#endif

	htab.destoryHashtab();
}

__host__ void construct_hashtab_soa_gpu (key_type * keys, value_type * values, size_t num_of_keys)
{
	hashtab_soa_t htab;
	key_type * dkeys = NULL;
	value_type * dvalues = NULL;
	CUDA_CHECK_RETURN (cudaMalloc (&dkeys, sizeof(key_type) * num_of_keys));
	CUDA_CHECK_RETURN (cudaMemcpy (dkeys, keys, sizeof(key_type) * num_of_keys, cudaMemcpyHostToDevice));
	if(values != NULL)
	{
		CUDA_CHECK_RETURN (cudaMalloc (&dvalues, sizeof(value_type) * num_of_keys));
		CUDA_CHECK_RETURN (cudaMemcpy (dvalues, values, sizeof(value_type) * num_of_keys, cudaMemcpyHostToDevice));
	}
	htab.createHashtab(num_of_keys, device_type::gpu);
	insert_key_value_hashtab_soa<<<MAX_NUM_BLOCKS, THREADS_PER_BLOCK>>>(dkeys, dvalues, num_of_keys);

#ifdef DEBUG
	size_t size = htab.getTabSize();
	printf ("primary table size::::: %lu\n", size);
	size = size * SLOT_WIDTH;
	int * hstates = (int *) malloc (sizeof(int) * size);
	key_type * hkeys = (key_type *) malloc (sizeof(key_type) * size);
	value_type * hvalues = (value_type *) malloc (sizeof(value_type) * size);
	CUDA_CHECK_RETURN (cudaMemcpy(hstates, htab.getDeviceStates(), sizeof(int) * size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN (cudaMemcpy(hkeys, htab.getDeviceKeys(), sizeof(key_type) * size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN (cudaMemcpy(hvalues, htab.getDeviceValues(), sizeof(value_type) * size, cudaMemcpyDeviceToHost));
	check_hash_results_soa(hstates, hkeys, hvalues, size);
	free (hstates);
	free (hkeys);
	free (hvalues);
#endif

	htab.destoryHashtab();
}

int main (void)
{
	size_t N = 100000;
	uint * key = (uint *) malloc (sizeof(uint) * N);
	CHECK_PTR_RETURN (key, "malloc host keys error!\n");
	uint * value = (uint *) malloc (sizeof(uint) * N);
	CHECK_PTR_RETURN (value, "malloc host values error!\n");

	for (size_t i=0; i<N; i++)
	{
		key[i] = i+1;
		value[i] = key[i]+1;
	}
	construct_hashtab_gpu (key, value, N);
//	construct_hashtab_soa_gpu (key, value, N);
	construct_hashtab_warp_gpu (key, value, N);
	uint a = 0b10000000000000000000000000000000;
	for (int t=0; t<32; t++)
	{
		int r;
	//	r = HMultiplyDeBruijnBitPosition[((uint)((a & -a) * 0x077CB531U)) >> 27];
	//	printf ("bit set = %d, 1\n", r);
	}

	free(key);
	free(value);
}
}
#endif /*HASH_CUH*/
