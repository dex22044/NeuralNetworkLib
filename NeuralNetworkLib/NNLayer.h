#pragma once

#include <cstdlib>

class NNLayer
{
public:
	int size;
	double* neurons;
	double* biases;
	double** weights;

	NNLayer(int size, int nextSize);
};

