/*
 * Container.cpp
 *
 *  Created on: 2019-2-11
 *      Author: qiushuang
 */

#include <iostream>
#include "Container.h"

using namespace std;

void use (Container& c)
{
	const int sz = c.size();
	for (int i=0; i!=sz; ++i)
		cout << c[i] << '\n';
}

void testVector()
{
	double arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	Vector_container vc(9, arr);
	use(vc);
}

void testList()
{
	List_container lc {9, 8, 7, 6, 5, 4, 3, 2, 1};
	use(lc);
}

void testVectorDynamic()
{
	double arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	Vector_dynamic q(9, arr);
	const int sz = q.size();
	for (int i=0; i!=sz; ++i)
		cout << q[i] << '\n';
}

int main (void)
{
	testVector();
	testList();
	testVectorDynamic();

	return 0;
}
