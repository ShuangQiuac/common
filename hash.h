/*
 * hash.h
 *
 *  Created on: 2015-7-9
 *      Author: qiushuang
 */

#ifndef HASH_H_
#define HASH_H_


#include "utility.h"

#define ELEM_FACTOR 0.9 // 0.27 for hg7, 0.25 for human14 and ecoli

#define EMPTY // empty entry slot
#define MAX_TABLE_SIZE 4294967295 // 2^32-1
#define DEFAULT_SEED 3735928559

#define THREADS_PER_TABLE 24 // threads in cpu

typedef uint hashsize_t;
typedef uint hashval_t;
typedef uint prime_index_t;


/* MURMUR HASH FUNCTION: */

#define ONE32   0xFFFFFFFFUL
#define LL(v)   (v##ULL)
#define ONE64   LL(0xFFFFFFFFFFFFFFFF)

#define T32(x)  ((x) & ONE32)
#define T64(x)  ((x) & ONE64)

#define ROTL32(v, n)   \
	(T32((v) << ((n)&0x1F)) | T32((v) >> (32 - ((n)&0x1F))))

#define ROTL64(v, n)   \
	(T64((v) << ((n)&0x3F)) | T64((v) >> (64 - ((n)&0x3F))))

#define UNIT_LENGTH 1
#define UNIT_BYTES 4

__device__ __host__ static uint
murmur_hash3_32 (const uint * key, uint seed)
{
	  int i;

	  uint h = seed;
	  uint k;

	  uint c1 = 0xcc9e2d51;
	  uint c2 = 0x1b873593;

	  for (i = 0; i < UNIT_LENGTH; i++)
	  {
		  k = *key++;
		  k *= c1;
		  k = ROTL32(k,15);
		  k *= c2;
		  h ^= k;
		  h = ROTL32(h,13);
		  h = h*5+0xe6546b64;
	  }

	  h ^= (UNIT_LENGTH * UNIT_BYTES);

	  h ^= h >> 16;
	  h *= 0x85ebca6b;
	  h ^= h >> 13;
	  h *= 0xc2b2ae35;
	  h ^= h >> 16;

	  return h;

}


/*FNV HASH FUNCTION*/

#define FNV_PRIME_32 16777619UL
#define OFFSET_BASIS_32 2166136261UL

__device__ __host__ static uint fnv_hash_32 (const char * key)
{
	uint h = OFFSET_BASIS_32;
	for (int i=0; i<UNIT_BYTES; i++)
	{
		h ^= *key++;
		h *= FNV_PRIME_32;
	}

	return h;
}


/*BIT OPERATIONS, reference: http://graphics.stanford.edu/~seander/bithacks.html#SelectPosFromMSBRank*/
#define BYTE_BIT 8
#define count_bit_set(v, pos, r) {\
		  r = v >> (sizeof(v) * BYTE_BIT - pos); \
		  r = r - ((r >> 1) & ~0UL/3); \
		  r = (r & ~0UL/5) + ((r >> 2) & ~0UL/5); \
		  r = (r + (r >> 4)) & ~0UL/17; \
		  r = (r * (~0UL/255)) >> ((sizeof(v) - 1) * BYTE_BIT); }


#endif /* HASH_H_ */
