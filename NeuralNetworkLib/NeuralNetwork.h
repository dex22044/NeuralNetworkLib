#pragma once

#include <string>

#include "NNLayer.h"
#include "NNTools.h"

class NeuralNetwork
{
public:
	double learningRate;
	int layers;
	NNLayer** nnlayers;

	NeuralNetwork(double learningRate, int layers, int* sizes);

	double* FeedForward(double* inputData);
	void Backpropogation(double* targets);
};

