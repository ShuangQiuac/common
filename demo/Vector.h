/*
 * Vector.h
 *
 *  Created on: 2019-2-11
 *      Author: qiushuang
 */

#ifndef VECTOR_H_
#define VECTOR_H_

#include <string>
using namespace std;

class Vector {
public:
	Vector(int, double*);
	// declaration of the constructor
	~Vector();
	// declaration of the destructor
	double& operator[](int);
	// declaration of subscripting
	int size();
	void cout(string);
	// declaration of the function to get Vector size
protected:
	double* elem;
	// elem points to an array of sz doubles
	int sz;
	// number of elements in Vector
};

class Vector_dynamic : public Vector {
// subclass derived from Vector
public:
	Vector_dynamic(int, double*);
	// constructor
	~Vector_dynamic();
	// destructor
	double pushBack(double elem);
	// a new method to append an element
private:
	int index;
	// index to push a new element
};
#endif /* VECTOR_H_ */
