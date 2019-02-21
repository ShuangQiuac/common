/*
 * Vector.cpp
 *
 *  Created on: 2019-2-11
 *      Author: qiushuang
 */

#include <iostream>
#include <cstring>
#include <algorithm>
#include "Vector.h"

// assume s is the size of arr in our demo
Vector::Vector(int s, double * arr) // definition of the constructor
{
	elem = new double[s]; // initialize data member
	sz = s; // initialize data member
	for (int i=0; i!=sz; ++i) // initialize elements
		elem[i] = arr[i];
	std::cout << "Calling constructor" << std::endl;
}
Vector::~Vector() // definition of the destructor
{
	delete[] elem;
	std::cout << "Calling destructor" << std::endl;
}
double& Vector::operator[](int i) // definition of operator subscripting
{
	return elem[i];
}
int Vector::size() // definition of size()
{
	return sz;
}
void Vector::cout(string str) // definition of cout()
{
	std::cout << str << std::endl;
}

Vector_dynamic::Vector_dynamic(int s, double * arr) // definition of the constructor
: Vector(s, arr)
{
	if (arr == NULL)
		index = 0;
	else
		index = s;
}

Vector_dynamic::~Vector_dynamic() {}

// The implementation that will cause append error when index>=sz, comment this to demo correct implementation
double Vector_dynamic::pushBack(double element)
{
	elem[index++] = element;
	return elem[index-1];
}
// The implementation that resolve illegal memory access, comment this to demo incorrect implementation
/*
double Vector_dynamic::pushBack(double element)
{
	if (index >= sz) {
		double * newElem = new double[sz*2];
		std::memcpy(newElem, elem, sz*sizeof(double));
		delete[] elem;
		elem = newElem;
		sz *= 2;
	}
	elem[index++] = element;
	return elem[index-1];
}
*/
