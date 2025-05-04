# Neural Network from scratch in C++

This project implements a basic feedforward neural network using standard C++ libraries.
It implements forward propagation, backpropagation, and training with gradient descent on the classic XOR problem.

## Future developements
* Implement Layer abstraction
* Add other activation functions
* Implement a Tensor class
* Solve classification problems

## Build & Run

### Requirements
* C++17-compatible compiler

### Steps

```bash
git clone https://github.com/RomanJanovcik/nn-from-scratch
cd nn-from-scratch
mkdir build
cd build
cmake ..
make
./nn_from_scratch
```

### For testing (e.g. test_xor)
```bash
make test_xor
./test_xor
