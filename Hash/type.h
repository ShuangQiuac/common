/*
 * type.h
 *
 *  Created on: 2019-2-22
 *      Author: qiushuang
 */

#ifndef TYPE_H_
#define TYPE_H_

#include <stdint.h>
#include <stddef.h>

#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif

typedef unsigned long long ull;
typedef unsigned long long idtype;
typedef unsigned char uch;
typedef uint32_t uint;
typedef uint offset_t;

typedef struct timeval evaltime_t;

typedef uint hashsize_t;
typedef uint hashval_t;
typedef uint prime_index_t;

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


#define THREADS_PER_BLOCK 1024
#define MAX_NUM_BLOCKS 64 // maximum number of blocks
#define MAX_NUM_THREADS (MAX_NUM_BLOCKS*THREADS_PER_BLOCK)
#define NUM_WARPS_PER_BLOCK 32
#define WARP_WIDTH 32
#define SLOT_WIDTH 32 // number of slots per warp
#define FULL_MASK 0xffffffff
#define BYTE_BIT 8

#endif /* TYPE_H_ */
