/*
 * test.cpp
 *
 *  Created on: 2019-2-22
 *      Author: qiushuang
 */
#include "utility.h"
#include "test.h"

void TestDriver::check_hash_results (entry_t * tab, size_t size)
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

void TestDriver::check_hash_results_soa (int * states, key_type * keys, value_type * values, size_t size)
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
		fprintf (tabfile, "%lu, %lu\n", i+1, i+2);
	}
	fclose(tabfile);
}

void TestDriver::check_hash_results_warp (entry_kv_t * tab, size_t size)
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
			ptr = (char*)ptr + sizeof(state_type);
			slot_mask_type slot_mask = *(slot_mask_type*)ptr;
			ptr = (char*)ptr + sizeof(slot_mask_type);
			for (int j=0; j<SLOT_WIDTH; j++)
			{
				if(slot_mask & (1 << (sizeof(slot_mask)*BYTE_BIT - j - 1)))
				{
					count++;
//					fprintf (tabfile, "%u, %u\n", ((entry_kv_t*)ptr)->key, ((entry_kv_t*)ptr)->value);
				}
				ptr = (char*)ptr + sizeof(entry_kv_t);
			}
		}
	}
	printf ("NUMBER OF ELEMENTS IN HASH TABLE: %lu\n", count);
	fclose(tabfile);
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
	TestDriver test;
	test.construct_hashtab_gpu (key, value, N);
//	test.construct_hashtab_soa_gpu (key, value, N);
	test.construct_hashtab_warp_gpu (key, value, N);
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
