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
            double val = data(i, j);
            // Avoid overflow in exponentiation
            if (val > 0) {
                result(i, j) = 1.0 / (1.0 + exp(-val));
            } else {
                result(i, j) = exp(val) / (1.0 + exp(val));
            }
        }
    }

    return result;
}

Matrix sigmoid_derivative(const Matrix& data) {
	unsigned rows = data.getRows();
	unsigned cols = data.getCols();	

	Matrix result(rows, cols);
    Matrix sig = sigmoid(data);

	for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
			result(i, j) = sig(i, j) * (1 - sig(i, j));
        }
    }

	return result;
}

// Mean square error
double mse(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Size mismatch between y_true and y_pred");
    }

    double sum = 0.0;
    for (unsigned i = 0; i < y_true.size(); ++i) {
        double diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return sum / y_true.size();
}
