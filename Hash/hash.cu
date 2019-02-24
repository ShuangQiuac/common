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
#include <cooperative_groups.h>
#include "hash.h"
#include "hashtab.h"
#include "test.h"
#include "hashtab.cuh"

#define DEBUG

#ifndef HASH_CUH
#define HASH_CUH

using namespace cooperative_groups;

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
		printf ("Error: index greater than hash table size!\n");
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
//	uint laneid = tid % WARP_WIDTH;
	thread_block_tile<4> tile4 = tiled_partition<4>(this_thread_block());
	uint laneid = tile4.thread_rank();

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

__global__ void insert_key_value_hashtab_cooperative_group (key_type * input_keys, value_type * input_values, size_t num_of_keys)
{
	uint gid = blockDim.x * blockIdx.x + threadIdx.x;
	uint tid = threadIdx.x;
	int warpsize = 8;
	int slotsize = 8;
	uint warpid = gid / warpsize;
	thread_block_tile<4> tile4 = tiled_partition<4>(this_thread_block());
	uint laneid = tile4.thread_rank();
	ull full_mask;
	if (warpsize == 8)
		full_mask = 0xff000000;
	else if (warpsize == 16)
		full_mask = 0xffff0000;
	else if (warpsize == 32)
		full_mask = 0xffffffff;
	else if (warpsize == 4)
		full_mask = 0xf0000000;

	size_t keys_per_warp = (num_of_keys + (MAX_NUM_THREADS/warpsize - 1)) / (MAX_NUM_THREADS/warpsize); // keys per block
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
		ptr = (void*)((char*)entries + index1 * (sizeof(entry_kv_t) * slotsize + sizeof(slot_mask_type) + sizeof(state_type)));
		state_ptr = (state_type *)ptr;
		slot_mask_ptr = (slot_mask_type*)((char*)state_ptr + sizeof(state_type));
		entry = (entry_kv_t*)((char*)slot_mask_ptr + sizeof(slot_mask_type) + sizeof(entry_kv_t) * laneid);

		while (*slot_mask_ptr != full_mask)
		{
			if (laneid == 0)
			{
				flag = atomicCAS(state_ptr, 0, 1);
			}
			flag = __shfl_sync (full_mask, flag, 0, warpsize); // broadcast flag to threads in this warp
			active = *slot_mask_ptr;
			if (flag == 0) // empty slot exists
			{
				uint r = MultiplyDeBruijnBitPosition[((uint)(((~active) & -(~active)) * 0x077CB531U)) >> 27];
				if ((~active) && r == warpsize - 1 - laneid) // thread selected for insertion
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
				cont_flag = __shfl_sync (full_mask, cont_flag, warpsize-1-r, warpsize);
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
			flag = __shfl_sync (full_mask, flag, 0, warpsize); // broadcast flag to threads in this warp
			int ret = 0;
			ret = compare_keys (key, &(entry->key));
			uint ballot_result = __ballot_sync (full_mask, ret);
			uint r = MultiplyDeBruijnBitPosition[((uint)((ballot_result & -ballot_result) * 0x077CB531U)) >> 27];
			if (ballot_result && r == warpsize - 1 - laneid)
			{
				replace_int_value<value_type> (&(entry->value), *value);
				cont_flag = 1;
			}
			cont_flag = __shfl_sync (full_mask, cont_flag, warpsize-1-r, warpsize);
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

			ptr = (void*)((char*)entries + index1 * (sizeof(entry_kv_t) * warpsize + sizeof(slot_mask_type) + sizeof(state_type)));
			state_ptr = (state_type *)ptr;
			slot_mask_ptr = (slot_mask_type*)((char*)state_ptr + sizeof(state_type));
			entry = (entry_kv_t*)((char*)slot_mask_ptr + sizeof(slot_mask_type) + sizeof(entry_kv_t) * laneid);
			while (*slot_mask_ptr != full_mask)
			{
				if (laneid == 0)
				{
					flag = atomicCAS(state_ptr, 0, 1);
				}
				flag = __shfl_sync (full_mask, flag, 0, warpsize); // broadcast flag to threads in this warp
				active = *slot_mask_ptr;
				if (flag == 0) // empty slot exists
				{
					uint r = MultiplyDeBruijnBitPosition[((uint)(((~active) & -(~active)) * 0x077CB531U)) >> 27];
					if ((~active) && r == warpsize - 1 - laneid) // thread selected for insertion
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
					cont_flag = __shfl_sync (full_mask, cont_flag, warpsize-1-r, warpsize);
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
				flag = __shfl_sync (full_mask, flag, 0, warpsize); // broadcast flag to threads in this warp
				int ret = 0;
				ret = compare_keys (key, &(entry->key));
				uint ballot_result = __ballot_sync (full_mask, ret);
				uint r = MultiplyDeBruijnBitPosition[((uint)((ballot_result & -ballot_result) * 0x077CB531U)) >> 27];
				if (ballot_result && r == warpsize - 1 - laneid)
				{
					replace_int_value<value_type> (&(entry->value), *value);
					cont_flag = 1;
				}
				cont_flag = __shfl_sync (full_mask, cont_flag, warpsize-1-r, warpsize);
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


void TestDriver::construct_hashtab_gpu (key_type * keys, value_type * values, size_t num_of_keys)
{
	HashTab htab;
	key_type * dkeys = NULL;
	value_type * dvalues = NULL;
	CUDA_CHECK_RETURN (cudaMalloc (&dkeys, sizeof(key_type) * num_of_keys));
	CUDA_CHECK_RETURN (cudaMemcpy (dkeys, keys, sizeof(key_type) * num_of_keys, cudaMemcpyHostToDevice));
	if(values != NULL)
	{
		CUDA_CHECK_RETURN (cudaMalloc (&dvalues, sizeof(value_type) * num_of_keys));
		CUDA_CHECK_RETURN (cudaMemcpy (dvalues, values, sizeof(value_type) * num_of_keys, cudaMemcpyHostToDevice));
	}
	evaltime_t start, end;
	float hashInsertTime = 0;
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
	htab.createHashtab(num_of_keys, device_type::gpu);
	insert_key_value_hashtab<<<MAX_NUM_BLOCKS, THREADS_PER_BLOCK>>>(dkeys, dvalues, num_of_keys);
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	hashInsertTime = (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	printf ("Hash insert  time: %f\n", hashInsertTime);

#ifdef DEBUG
	size_t size = htab.getTabSize();
	printf ("primary table size::::: %lu\n", size);
	size = size * SLOT_WIDTH;
	entry_t * table = (entry_t *) malloc (sizeof(entry_t) * size);
	CUDA_CHECK_RETURN (cudaMemcpy(table, htab.getDeviceTab(), sizeof(entry_t) * size, cudaMemcpyDeviceToHost));
	TestDriver test;
	test.check_hash_results(table, size);
	free (table);
#endif

	htab.destoryHashtab();
}

void TestDriver::construct_hashtab_warp_gpu (key_type * keys, value_type * values, size_t num_of_keys)
{
	HashTabKv htab;
	key_type * dkeys = NULL;
	value_type * dvalues = NULL;
	CUDA_CHECK_RETURN (cudaMalloc (&dkeys, sizeof(key_type) * num_of_keys));
	CUDA_CHECK_RETURN (cudaMemcpy (dkeys, keys, sizeof(key_type) * num_of_keys, cudaMemcpyHostToDevice));
	if(values != NULL)
	{
		CUDA_CHECK_RETURN (cudaMalloc (&dvalues, sizeof(value_type) * num_of_keys));
		CUDA_CHECK_RETURN (cudaMemcpy (dvalues, values, sizeof(value_type) * num_of_keys, cudaMemcpyHostToDevice));
	}
	evaltime_t start, end;
	float hashInsertTime = 0;
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
	htab.createHashtab(num_of_keys, device_type::gpu);
//	insert_key_value_hashtab_warp<<<MAX_NUM_BLOCKS, THREADS_PER_BLOCK>>>(dkeys, dvalues, num_of_keys);
	insert_key_value_hashtab_cooperative_group<<<MAX_NUM_BLOCKS, THREADS_PER_BLOCK>>>(dkeys, dvalues, num_of_keys);
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	hashInsertTime = (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	printf ("Hash insert  time: %f\n", hashInsertTime);

#ifdef DEBUG
	size_t size = htab.getTabSize();
	printf ("primary table size::::: %lu\n", size);
	entry_kv_t * table = (entry_kv_t *) malloc ((sizeof(entry_kv_t) * SLOT_WIDTH + sizeof(slot_mask_type) + sizeof(state_type)) * size);
	CUDA_CHECK_RETURN (cudaMemcpy(table, htab.getDeviceTab(), (sizeof(entry_kv_t) * SLOT_WIDTH + sizeof(slot_mask_type) + sizeof(state_type)) * size, cudaMemcpyDeviceToHost));
	TestDriver test;
	test.check_hash_results_warp(table, size);
	free (table);
#endif

	htab.destoryHashtab();
}

void TestDriver::construct_hashtab_soa_gpu (key_type * keys, value_type * values, size_t num_of_keys)
{
	HashTabSoa htab;
	key_type * dkeys = NULL;
	value_type * dvalues = NULL;
	CUDA_CHECK_RETURN (cudaMalloc (&dkeys, sizeof(key_type) * num_of_keys));
	CUDA_CHECK_RETURN (cudaMemcpy (dkeys, keys, sizeof(key_type) * num_of_keys, cudaMemcpyHostToDevice));
	if(values != NULL)
	{
		CUDA_CHECK_RETURN (cudaMalloc (&dvalues, sizeof(value_type) * num_of_keys));
		CUDA_CHECK_RETURN (cudaMemcpy (dvalues, values, sizeof(value_type) * num_of_keys, cudaMemcpyHostToDevice));
	}
	evaltime_t start, end;
	float hashInsertTime = 0;
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&start, NULL);
	htab.createHashtab(num_of_keys, device_type::gpu);
	insert_key_value_hashtab_soa<<<MAX_NUM_BLOCKS, THREADS_PER_BLOCK>>>(dkeys, dvalues, num_of_keys);
	CUDA_CHECK_RETURN (cudaDeviceSynchronize());
	gettimeofday (&end, NULL);
	hashInsertTime = (float)((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000;
	printf ("Hash insert  time: %f\n", hashInsertTime);

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
	TestDriver test;
	test.check_hash_results_soa(hstates, hkeys, hvalues, size);
	free (hstates);
	free (hkeys);
	free (hvalues);
#endif

	htab.destoryHashtab();
}

#endif /*HASH_CUH*/
