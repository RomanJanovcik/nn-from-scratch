#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "linalg.hpp"

// Template for a simple feed-forward neural network 
class NeuralNetwork {
	private:
		unsigned input_size, hidden_size, output_size;
		double learning_rate;
		Matrix w1, w2;
		Vector b1, b2;

	public:
		// Constructor
		NeuralNetwork(unsigned input_size, unsigned hidden_size, unsigned output_size, double learning_rate);

		// Feed forward 
		Vector forward(const Matrix& data);
		
		// Backpropagation
		void backward(const Matrix& data, const Matrix& y_true);

		// Update network hyperparameters
		void train(const Matrix& data, const Matrix& y_true, unsigned epochs);	
};

#endif // NEURAL_NETWORK_H
