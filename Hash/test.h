/*
 * test.h
 *
 *  Created on: 2019-2-22
 *      Author: qiushuang
 */

#ifndef TEST_H_
#define TEST_H_

#include "type.h"

class TestDriver {
public:
	// call kernels on gpu:
	void construct_hashtab_gpu (key_type *, value_type *, size_t);
	void construct_hashtab_warp_gpu (key_type *, value_type *, size_t);
	void construct_hashtab_soa_gpu (key_type *, value_type *, size_t);
	// check results:
	void check_hash_results (entry_t *, size_t size);
	void check_hash_results_soa (int *, key_type *, value_type *, size_t);
	void check_hash_results_warp (entry_kv_t *, size_t);
};


#endif /* TEST_H_ */
