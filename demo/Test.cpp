/*
 * User.cpp
 *
 *  Created on: 2019-2-11
 *      Author: qiushuang
 */

#include <iostream>
#include "Vector.h" // get Vector's interface

using namespace std;

// memory leak
void leak ()
{
	double * arr = new double[9];
	for (int i=1; i<10; ++i)
		arr[i-1] = i;
	Vector v(9, arr);
	for (int i=1; i<10; ++i)
		cout << v[i-1] << '\t';
	cout << endl;
}
// illegal memory access
void push ()
{
	double arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	Vector_dynamic q(9, arr);
	double qElem = q.pushBack(10);
	cout << "get pushed elem: " << qElem << endl;
}
// invalid memory access
Vector* access ()
{
	double arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	Vector v(9, arr);
	Vector* ref = &v;
	cout << "Within Vector scope: vector size = " \
			<< ref->size() << endl;
	return ref;
}


int main (void)
{
	cout << "Test leak:" << endl;
	leak();
	cout << "Test scope of access:" << endl;
	Vector* ret = access();
	cout << "Test push value:" << endl;
	push();
	cout << "Out of Vector scope: vector size = " << ret->size() << endl;
	return 0;
}
