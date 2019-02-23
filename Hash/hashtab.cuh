/*
 * hashtab.cuh
 *
 *  Created on: 2019-2-22
 *      Author: qiushuang
 */
#include "hashtab.h"

__constant__ static entry_t * dtab; // hash table pointer on the GPU
__constant__ static size_t dsize; // hash table size on the GPU
__constant__ static prime_index_t d_size_prime_index; // size prime index on the GPU
//* if using structure of arrays hash table: *
__constant__ static int * state_flags;
__constant__ static key_type * keys;
__constant__ static value_type * values;
//* using warp-centric hash table probing
__constant__ static entry_kv_t * htable;

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

void HashTab::createHashtab (size_t num_of_elems, device_type d)
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

void HashTab::destoryHashtab (void)
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

entry_t * HashTab::getDeviceTab (void) { return tab; }
size_t HashTab::getTabSize (void) { return size; }


void HashTabSoa::createHashtab (size_t num_of_elems, device_type d)
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

void HashTabSoa::destoryHashtab (void)
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

key_type * HashTabSoa::getDeviceKeys (void) { return tab.keys; }
value_type * HashTabSoa::getDeviceValues (void) { return tab.values; }
int * HashTabSoa::getDeviceStates (void) { return tab.state_flags; }
size_t HashTabSoa::getTabSize (void) { return size; }


void HashTabKv::createHashtab (size_t num_of_elems, device_type d)
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

void HashTabKv::destoryHashtab (void)
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

entry_kv_t * HashTabKv::getDeviceTab (void) { return tab; }
size_t HashTabKv::getTabSize (void) { return size; }
