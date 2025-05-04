#ifndef LINALG_H
#define LINALG_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>

class Vector {
private:
    std::vector<double> elements;

public:
    // Constructors
    Vector() {}
    Vector(unsigned size) : elements(size) {}
    Vector(const std::vector<double>& values) : elements(values) {}

    // Access
    unsigned size() const { return elements.size(); }
    double& operator[](int i) { return elements[i]; }
    const double& operator[](int i) const { return elements[i]; }

    // Vector addition
    Vector operator+(const Vector& other) const {
        if (size() != other.size()) throw std::invalid_argument("Size mismatch");
        Vector result(size());
        for (int i = 0; i < size(); ++i) {
            result[i] = elements[i] + other[i];
        }
        return result;
    }

	// Vector subtraction
	Vector operator-(const Vector& other) const {
		if (size() != other.size()) throw std::invalid_argument("Size mismatch");
		Vector result(size());
		for (int i=0; i < size(); ++i) {
			result[i] = elements[i] - other[i];
		}
		return result;
	}

    // Scalar multiplication
    Vector operator*(double scalar) const {
        Vector result(size());
        for (int i = 0; i < size(); ++i) {
            result[i] = elements[i] * scalar;
        }
        return result;
    }

	double sum() const {
		double result = 0.0;
		for (double val: elements) {
			result += val;
		}
		return result;
	}

	double mean() const {
		return sum()/size();
	}

    // Dot product
    double dot(const Vector& other) const {
        if (size() != other.size()) throw std::invalid_argument("Size mismatch");
        double sum = 0.0;
        for (int i = 0; i < size(); ++i) {
            sum += elements[i] * other[i];
        }
        return sum;
    }

	static Vector zeros(unsigned size) {
		Vector result(size);
		for (int i = 0; i < size; ++i) {
			result[i] = 0.0;
		}
		return result;
	}
	
    // Utility: Print vector
    void print() const {
        for (double val : elements) std::cout << val << " ";
        std::cout << std::endl;
    }
};

class Matrix {
private:
    std::vector<std::vector<double>> elements;
    unsigned rows, cols;

public:
    // Constructors
	Matrix() {};

    Matrix(unsigned rows, unsigned cols) : rows(rows), cols(cols) {
        elements.resize(rows, std::vector<double>(cols, 0.0));
    }

    Matrix(const std::vector<std::vector<double>>& values) {
        rows = values.size();
        cols = values.empty() ? 0 : values[0].size();
        elements = values;
    }

    // Access
    unsigned getRows() const { return rows; }
	unsigned getCols() const { return cols; }

    double& operator()(unsigned row, unsigned col) { return elements[row][col]; }
    const double& operator()(unsigned row, unsigned col) const { return elements[row][col]; }

    // Matrix addition
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix sizes must match for addition");
        }
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = elements[i][j] + other(i, j);
            }
        }
        return result;
    }

	// Add vector to each row
	Matrix operator+(const Vector& other) const {
		if (cols != other.size()) {
			throw std::invalid_argument("Number of columns has to equal vector size");
		}
		Matrix result(rows, cols);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				result(i, j) = elements[i][j] + other[j];
			}
		}
		return result;
	}

    // Matrix subtraction
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix sizes must match for addition");
        }
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = elements[i][j] - other(i, j);
            }
        }
        return result;
    }

    // Scalar multiplication
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = elements[i][j] * scalar;
            }
        }
        return result;
    }

	// Elemetwise multiplication 
    Matrix operator*(const Matrix& other) const {
		if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for elementwise multiplication");
        }
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = elements[i][j] * other(i, j);
            }
        }
        return result;
    }

    // Matrix multiplication (for simplicity, assumes compatible sizes)
    Matrix dot(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication");
        }
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                result(i, j) = 0;
                for (int k = 0; k < cols; ++k) {
                    result(i, j) += elements[i][k] * other(k, j);
                }
            }
        }
        return result;
    }
	
	Matrix transpose() const {
		Matrix result(cols, rows);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				result(j, i) = elements[i][j];
			}
		}
		return result;
	}

	Matrix sumRows() {
		Matrix result(rows, 1);
		for (int i = 0; i < rows; ++i) {
			double sum = 0.0;
			for (int j = 0; j < cols; ++j) {
				sum += elements[i][j];
			}
			result(i, 0) = sum;
		}
		return result;
	}

	Matrix sumCols() {
		Matrix result(1, cols);
		for (int j = 0; j < cols; ++j) {
			double sum = 0.0;
			for (int i = 0; i < rows; ++i) {
				sum += elements[i][j];
			}
			result(0, j) = sum;
		}
		return result;
	}

	static Matrix random(unsigned rows, unsigned cols, double min, double max) {
		Matrix result(rows, cols);
		std::random_device rd;
		std::mt19937 gen(rd());

		std::uniform_real_distribution<double> dist(min, max);

		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				result(i, j) = dist(gen);
			}
		}
		return result;
	}

	static Matrix zeros(unsigned rows, unsigned cols) {
		Matrix result(rows, cols);

		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				result(i, j) = 0.0;
			}
		}
		return result;
	}

	Vector toVector() const {
		if (rows == 1) {
			Vector result(cols);
			for (int j = 0; j < cols; ++j) {
				result[j] = elements[0][j];
			}
			return result;
		}

		else if(cols == 1) {
			Vector result(rows);
			for (int i = 0; i < rows; ++i) {
				result[i] = elements[i][0];
			}
			return result;	
		}

		else {
			throw std::invalid_argument("At least one dimension has to equal 1");
		}
		
	}
	
    // Utility: Print matrix
    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << elements[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

#endif // LINALG_H
