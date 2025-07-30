#include "funcs.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <functional>
#include <numeric> // for std::inner_product
#include <map>
#include <limits>
#include <pybind11/numpy.h>  // For working with NumPy arrays


namespace py = pybind11;


// Function signature with types
double relaxedFunction(
    const std::vector<double>& x, 
    std::function<double(const std::vector<double>&)> function, 
    double alpha, 
    const std::vector<double>& lb, 
    const std::vector<double>& ub
) {
    // Calculate the difference between the current point and the lower bound
    std::vector<double> diff_lb(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        diff_lb[i] = x[i] - lb[i];
    }

    // Calculate the difference between the upper bound and the current point
    std::vector<double> diff_ub(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        diff_ub[i] = ub[i] - x[i];
    }

    // Compute the dot product (inner product) of the differences.
    // This represents how far x is from both bounds.
    double dot_product = std::inner_product(diff_lb.begin(), diff_lb.end(), diff_ub.begin(), 0.0);

    // Evaluate the original function at point x
    double original_value = function(x);

    // Subtract the alpha-scaled dot product from the function value.
    // This relaxes the function by reducing it according to the bounds and alpha.
    double relaxed_value = original_value - alpha * dot_product;

    // Return the relaxed function value.
    return relaxed_value;
}


// C++ function to compute upper bounds using Python's scipy.optimize.minimize
std::vector<std::map<std::string, std::vector<double>>> computeUpperBounds(
    std::vector<std::map<std::string, std::vector<double>>>& nodes, 
    py::object objective) {

    double inf = std::numeric_limits<double>::infinity();

    for (auto& iNode : nodes) {
        // Assuming "ubd" is the value of interest in the vector
        if (iNode["ubd"].empty() || iNode["ubd"][0] == inf) {
            // Calculate the midpoint between lower and upper bounds
            std::vector<double> lb = iNode["lb"];
            std::vector<double> ub = iNode["ub"];
            std::vector<double> x0(lb.size());

            for (size_t i = 0; i < lb.size(); ++i) {
                x0[i] = (lb[i] + ub[i]) / 2.0;
            }

            // Create bounds for scipy.optimize.minimize
            py::list bnds;
            for (size_t i = 0; i < lb.size(); ++i) {
                bnds.append(py::make_tuple(lb[i], ub[i]));
            }

            // Call Python's scipy.optimize.minimize
            py::object scipy_minimize = py::module_::import("scipy.optimize").attr("minimize");
            py::object solUBD = scipy_minimize(objective, x0, py::arg("bounds") = bnds, py::arg("method") = "L-BFGS-B");

            // Update the node with the computed upper bound
            iNode["ubd"][0] = solUBD.attr("fun").cast<double>();  // Update the first element of "ubd"
        }
    }

    return nodes;
}