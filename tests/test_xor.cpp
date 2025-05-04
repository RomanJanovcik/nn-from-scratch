#include "../include/neural_network.hpp"
#include <cassert>
#include <iostream>

void test_xor() {
	std::cout<<"Testing XOR accuracy:"<<std::endl;
	NeuralNetwork nn_xor(2, 4, 1, 0.1);

	Matrix data({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
	Matrix y_true({{0}, {1}, {1}, {0}});

	nn_xor.train(data, y_true, 10000);

	Vector y_pred = nn_xor.forward(data);

	assert(y_pred[0] < 0.1);
	assert(y_pred[1] > 0.9);
	assert(y_pred[2] > 0.9);
	assert(y_pred[3] < 0.1);

	std::cout << "XOR test passed." << std::endl;	
}

int main(){
	test_xor();
	return 0;
}

