# Micrograd-C

**Micrograd-C** is a simple automatic differentiation library implemented in C, basically a 'C' version of Karpathy's [Micrograd](https://github.com/karpathy/micrograd). It provides a basic framework for creating and manipulating scalar values with gradient support, allowing for backpropagation through a computation graph.

## Features

- **Automatic Differentiation**: Compute gradients for scalar operations using backpropagation.
- **Scalar Operations**: Supports addition, multiplication, power, ReLU, sigmoid, and tanh functions.
- **Backward Pass**: Propagate gradients through a computation graph.

## Getting Started

To get started with Micrograd-C, clone this repository and compile the provided code.

### Prerequisites

- A C compiler (e.g., `gcc`)
- A C++ compiler (e.g. `g++`)

### Building the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/shivendrra/micrograd.c.git
   cd micrograd-c
   ```

2. Compile the code:
- For c version
   ```bash
   gcc csrc/scalar.c test.c -o test -lm
   ```
- For cpp version
   ```bash
   g++ cppsrc/value.cpp cppsrc/module.cpp  main.cpp -o main -lm
   ```

### Running the Example

After compiling, run the example:

- For c version
```bash
./test
```

- For cpp version
```bash
./main
```

### Example Output

Here is an example of what you can expect when running the program:

```
Scalar[data=(2), grad=0.0074]
Scalar[data=(3), grad=0.0049]
Scalar[data=(5), grad=0]
Scalar[data=(6), grad=0.0025]
Scalar[data=(6), grad=0]
Scalar[data=(1), grad=0]
Scalar[data=(0.9975), grad=1]
```

## Code Overview

### `scalar.h` / `value.h`

Header file defining the `Scalar` structure and function prototypes for scalar operations and backward propagation.

### `scalar.c` / `value.cpp`

C file containing all the necessary functions for `Scalar` value structure for ops & backprop.

### `nn.h` / `module.h`

Header file defining the `Neuron`, `Layer`, & `MLP` structures & function prototypes for creating a small MLP.

### `nn.c` / `module.cpp`

C file containing all the basic functions for building & implementing MLP in C using `Scalar` values.

### `test.c` / `cppsrc/main.cpp`

Demonstrates the usage of the Micrograd-C library by creating scalar values, performing operations, and computing gradients.

### Functions

- `initialize_scalars`: Initializes a scalar value with a given data value and its children.
- `add_val`, `mul_val`, `pow_val`, `relu`, `sigmoid`, `tan_h`, `sub_val`: Functions for scalar operations.
- `backward`: Computes gradients for all scalar values in the computation graph.
- `print`: Prints the scalar data and gradient values.

## Contributing

Feel free to contribute by submitting issues or pull requests. Your contributions are welcome!

## License

None!