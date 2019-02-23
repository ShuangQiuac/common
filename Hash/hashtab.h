/*
 * hash_table.h
 *
 *  Created on: 2019-2-22
 *      Author: qiushuang
 */

#ifndef HASH_TABLE_H_
#define HASH_TABLE_H_

#include "type.h"

class HashTab
{
private:
	entry_t * tab; // hash table pointer, either point to device or point to a host hash table
	size_t size; // hash table size
	prime_index_t size_prime_index; // size prime index on the host
	device_type device; // CPU or GPU

public:
	void createHashtab (size_t, device_type);

	void destoryHashtab (void);

	entry_t * getDeviceTab (void);
	size_t getTabSize (void);
};

class HashTabSoa
{
private:
	entry_soa_t tab; // structure of arrays of hash table, accessible only on the host
	size_t size; // hash table size
	prime_index_t size_prime_index; // size prime index on the host
	device_type device; // CPU or GPU

public:
	void createHashtab (size_t, device_type);

	void destoryHashtab (void);

	key_type * getDeviceKeys (void);
	value_type * getDeviceValues (void);
	int * getDeviceStates (void);
	size_t getTabSize (void);
};


class HashTabKv
{
private:
	entry_kv_t * tab; // hash table pointer, either point to device or point to a host hash table
	size_t size; // hash table size
	prime_index_t size_prime_index; // size prime index on the host
	device_type device; // CPU or GPU

public:
	void createHashtab (size_t, device_type);

	void destoryHashtab (void);

	entry_kv_t * getDeviceTab (void);
	size_t getTabSize (void);
};


#endif /* HASH_TABLE_H_ */
