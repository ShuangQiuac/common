/*
 * User.cpp
 *
 *  Created on: 2019-2-12
 *      Author: qiushuang
 */

#include <iostream>
#include "Vector.h" // get Vector's interface

void testVectorDynamic()
{
	double arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	Vector_dynamic q(9, arr);
	const int sz = q.size();
	for (int i=0; i!=sz; ++i)
		cout << q[i] << '\n';
	q.cout("End of testVectorDynamic");
}

int main (void)
{
	cout << "Test dynamic vector: " << endl;
	testVectorDynamic();

	return 0;
}
