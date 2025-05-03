#include "utils.hpp"
#include "linalg.hpp"
#include <cmath>

// Sigmoid activation function
Matrix sigmoid(const Matrix& data) {
    unsigned rows = data.getRows();
    unsigned cols = data.getCols();

    Matrix result(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = 1.0 / (1.0 + exp((-1.0) * data(i, j)));
        }
    }

    return result;
}

Matrix sigmoid_derivative(const Matrix& data) {
	unsigned rows = data.getRows();
	unsigned cols = data.getCols();	

	Matrix result(rows, cols);

	for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double sig = 1.0 / (1.0 + exp((-1.0) * data(i, j)));
			result(i, j) = sig * (1 - sig);
        }
    }

	return result;
}

// Mean square error
double mse(const Vector& y_true, const Vector& y_pred) {
	return y_pred.squared_distance(y_true) / y_pred.size();
}
