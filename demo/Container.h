/*
 * Container.h
 *
 *  Created on: 2019-2-11
 *      Author: qiushuang
 */

#ifndef CONTAINER_H_
#define CONTAINER_H_

#include <list>
#include <initializer_list>
#include "Vector.h"

class Container {
public:
	virtual double& operator[](int) = 0; // pure virtual function
	virtual int size() = 0; // const member function
	virtual ~Container() {} // destructor
};

class Vector_container : public Container {
	// Vector_container implements Container
	Vector v;
public:
	Vector_container(int s, double* arr) : v(s, arr) {}
	// Vector of s elements
	~Vector_container() {}
	// overrides the base class destructor ~Container(),
    // and implicitly invoke the destructor ~Vector()
	double& operator[](int i) {return v[i];}
	int size() {return v.size();}
};

class List_container : public Container {
	// List_container implements Container
	std::list<double> ld; // (standard-library) list of doubles
public:
	List_container() {} // empty list
	List_container(std::initializer_list<double> il):ld{il} {}
	~List_container() {}

	double& operator[](int i) {
		for (auto& x : ld) { // for each x in ld
			if (i==0) return x;
			--i;}
//		throw out_of_range("List container");
	}
	int size() {return ld.size();}
};

#endif /* CONTAINER_H_ */
