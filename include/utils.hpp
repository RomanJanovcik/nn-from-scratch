#ifndef UTILS_H
#define UTILS_H

#include "linalg.hpp"

Matrix sigmoid(const Matrix& data);

Matrix sigmoid_derivative(const Matrix& data);

double mse(const Vector& y_true, const Vector& y_pred);

#endif // UTILS_H
