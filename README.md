Optimization Tools

This repository contains a set of tools for numerical optimization, including classes for symbolic function wrapping, various optimization methods, and scripts for benchmarking.

Key Features

OptFunc

A class that acts as a symbolic wrapper for a function. It automatically computes the gradient and Hessian matrix if they are not provided manually, making it easier to work with complex functions.

LinSearch

This class implements three common optimization methods:

BFGS (Broyden–Fletcher–Goldfarb–Shanno)

Newton's Method

Gradient Descent

main.py: A simple script that demonstrates a basic implementation of the optimization classes.

benchmark.py: This script is designed to test the performance and robustness of the implemented optimization methods. It optimizes three well-known test functions:

Rosenbrock

Ackley

Griewank

The script measures the Mean Squared Error (MSE) and robustness for each method, and also generates a plot showing the best path to the solution found during the optimization process.
