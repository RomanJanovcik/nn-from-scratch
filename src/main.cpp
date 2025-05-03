#include "../include/linalg.hpp"
#include "../include/neural_network.hpp"
#include <iostream>

int main(){
	// Input and output values of an XOR gate
	Matrix data({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
	Matrix y_true({{0}, {1}, {1}, {0}});
	
	NeuralNetwork nn_xor(2, 4, 1, 0.1);

	nn_xor.train(data, y_true, 10000);

	Vector predictions = nn_xor.forward(data);

	std::cout << "\nPredictions:\n";
	predictions.print();	

	std::cout << "\nTrue values:\n";
	y_true.toVector().print();

	return 0;
}
