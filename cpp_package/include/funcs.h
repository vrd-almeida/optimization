#ifndef FUNCS_H
#define FUNCS_H

#include <vector>
#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <cmath>
#include <map>

// Function to calculate the gradient of a C++ function (independent of Python)
std::vector<double> gradient(const std::function<double(const std::vector<double>&)>& func, const std::vector<double>& x);

std::vector<std::vector<double>> hessian(const std::function<double(const std::vector<double>&)>& func, const std::vector<double>& x);

double round_to(double value, int decimal_places);

double adaptive_step(double x, double epsilon = 1e-7);

double dot_product(const std::vector<double>& v1, const std::vector<double>& v2);

// Struct to hold the result of the Steepest Descent algorithm
struct SteepestDescentResult {
    std::vector<double> xCur;               // Final point
    double fcur;                             // Function value at final point
    int iterations;                          // Number of iterations
    std::vector<std::vector<double>> xIter;  // History of iterates
    int ls_iter;                             // Total line search iterations
};

SteepestDescentResult steepest_descent(
    const std::function<double(const std::vector<double>&)>& func, 
    const std::vector<double>& x0, 
    const std::string& stepRule, 
    double c1, 
    double c2, 
    double alpha0, 
    double tol = 0.01
);

double relaxedFunction(
    const std::vector<double>& x, 
    std::function<double(const std::vector<double>&)> function, 
    double alpha, 
    const std::vector<double>& lb, 
    const std::vector<double>& ub
);

// Function declaration for computeUpperBounds
std::vector<std::map<std::string, std::vector<double>>> computeUpperBounds(
    std::vector<std::map<std::string, std::vector<double>>>& nodes, 
    pybind11::object objective);

#endif  // FUNCS_H
