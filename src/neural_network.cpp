#include "../include/linalg.hpp"
#include "../include/neural_network.hpp"
#include "../include/utils.hpp"
#include <iostream>

NeuralNetwork::NeuralNetwork(unsigned input_size, unsigned hidden_size, unsigned output_size, double learning_rate) : input_size(input_size), hidden_size(hidden_size), output_size(output_size), learning_rate(learning_rate) { 
	// Initial hyperparameter values
	w1 = Matrix::random(input_size, hidden_size, -1.0, 1.0);
	b1 = Vector::zeros(hidden_size);
	w2 = Matrix::random(hidden_size, output_size, -1.0, 1.0);
	b2 = Vector::zeros(output_size);
}

Vector NeuralNetwork::forward(const Matrix& data) {
	// Compute output of the network given an input
	Matrix output = sigmoid(sigmoid(data.dot(w1) + b1).dot(w2) + b2);
	return output.toVector();
}

void NeuralNetwork::backward(const Matrix& data, const Matrix& y_true) {
		// Feed forward
		Matrix hidden_activation = data.dot(w1) + b1;
		Matrix hidden_output = sigmoid(hidden_activation);

		Matrix output_activation = hidden_output.dot(w2) + b2;
		Matrix output = sigmoid(output_activation);

		// Backpropagation
		Matrix output_error = y_true - output;
		Matrix output_delta = output_error * sigmoid_derivative(output_activation);		
		
		Matrix d_w2 = hidden_output.transpose().dot(output_delta);
		Vector d_b2 = output_delta.sumCols().toVector();

		Matrix hidden_error = output_delta.dot(w2.transpose());
		Matrix hidden_delta = hidden_error * sigmoid_derivative(hidden_activation);

		Matrix d_w1 = data.transpose().dot(hidden_delta);
		Vector d_b1 = hidden_delta.sumCols().toVector();

		// Update hyperparameters
		w2 = w2 + (d_w2 * learning_rate); 
		b2 = b2 + (d_b2 * learning_rate);
		w1 = w1 + (d_w1 * learning_rate);
		b1 = b1 + (d_b1 * learning_rate);
}

void NeuralNetwork::train(const Matrix& data, const Matrix& y_true, unsigned epochs) {
	for (int epoch = 0; epoch < epochs; ++epoch) {
		// Calculate network output	
		Vector y_pred = forward(data);
		
		// Update network
		backward(data, y_true);

		// Compute loss
		double loss = mse(y_true.toVector(), y_pred);

		if (epoch % 1000 == 0) {
			std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
		}
	}
}
