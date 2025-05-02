#include <pybind11/pybind11.h>
#include <pybind11/stl.h>          // For automatic conversion of STL containers like std::vector
#include <pybind11/functional.h>   // For automatic conversion of std::function

#include "funcs.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_package, m) {
    m.doc() = R"pbdoc(
        My Optimization Module
        ----------------------

        This module provides functions for gradient calculation, Hessian calculation,
        and the Steepest Descent optimization method with support for various line search strategies.
    )pbdoc";

    // Bind the gradient function to Python
    m.def("gradient", &gradient,
        R"pbdoc(
            Calculate the gradient of a function.

            This function computes the gradient of a provided function at a specified point.
            The gradient is the vector of partial derivatives and points in the direction
            of the steepest ascent.

            Parameters:
                func (Callable[[List[float]], float]): A Python function that takes a list of floats as input and returns a float. This represents the function for which the gradient is to be calculated.
                x (List[float]): A list of floats representing the point at which the gradient should be calculated.

            Returns:
                List[float]: A list of floats representing the gradient of the function at the point x.
          )pbdoc",
        py::arg("func"), py::arg("x"));

    // Bind the hessian function to Python
    m.def("hessian", &hessian,
        R"pbdoc(
            Calculate the Hessian matrix of a function.

            This function computes the Hessian matrix of a provided function at a specified point.
            The Hessian matrix is a square matrix of second-order partial derivatives and provides
            information about the local curvature of the function.

            Parameters:
                func (Callable[[List[float]], float]): A Python function that takes a list of floats as input and returns a float. This represents the function for which the Hessian matrix is to be calculated.
                x (List[float]): A list of floats representing the point at which the Hessian matrix should be calculated.

            Returns:
                List[List[float]]: A list of lists representing the Hessian matrix of the function at the point x.
          )pbdoc",
        py::arg("func"), py::arg("x"));

    // Bind the SteepestDescentResult struct to Python
    py::class_<SteepestDescentResult>(m, "SteepestDescentResult",
        R"pbdoc(
            Result of the Steepest Descent optimization.

            This structure holds the result of the Steepest Descent algorithm, including
            the final position, function value at that position, number of iterations performed,
            the history of iterates, and the total number of line search iterations.
          )pbdoc")
        .def_readonly("xCur", &SteepestDescentResult::xCur,
            R"pbdoc(
                        The final position after optimization.

                        A list of floats representing the position in the search space where the
                        algorithm converged.
                      )pbdoc")
        .def_readonly("fcur", &SteepestDescentResult::fcur,
            R"pbdoc(
                        The function value at the final position.

                        A float representing the value of the function at the position `xCur`.
                      )pbdoc")
        .def_readonly("iterations", &SteepestDescentResult::iterations,
            R"pbdoc(
                        The number of iterations performed by the Steepest Descent algorithm.

                        An integer representing how many iterations were required for the algorithm
                        to converge.
                      )pbdoc")
        .def_readonly("xIter", &SteepestDescentResult::xIter,
            R"pbdoc(
                        The history of iterates during the optimization.

                        A list of lists of floats, where each inner list represents the position in
                        the search space at each iteration.
                      )pbdoc")
        .def_readonly("ls_iter", &SteepestDescentResult::ls_iter,
            R"pbdoc(
                        The total number of line search iterations.

                        An integer representing the cumulative number of iterations used by the line
                        search method across all Steepest Descent iterations.
                      )pbdoc");

    // Bind the steepest_descent function to Python
    m.def("steepest_descent", &steepest_descent,
        R"pbdoc(
            Perform the Steepest Descent optimization.

            This function implements the Steepest Descent method for minimizing a function.
            It supports different line search strategies, such as the Armijo rule, to determine
            the step size at each iteration.

            Parameters:
                function (Callable[[List[float]], float]): The objective function to be minimized. It should take a list of floats as input and return a float.
                x0 (List[float]): The initial guess for the optimization. A list of floats representing the starting point in the search space.
                stepRule (str): The rule to use for line search. Options are "armijo" for the Armijo rule.
                c1 (float): A parameter for the line search (e.g., sufficient decrease condition in the Armijo rule). Must be in the range (0, 1).
                c2 (float): A parameter for the line search (e.g., curvature condition in the Wolfe rule). Not currently used.
                alpha0 (float): The initial step size.
                tol (float, optional): The tolerance for the stopping condition. The algorithm stops when the gradient norm is less than `tol`. Default is 0.01.

            Returns:
                SteepestDescentResult: A structure containing the results of the optimization, including the final position, function value, number of iterations, history of iterates, and total line search iterations.
          )pbdoc",
        py::arg("function"), py::arg("x0"), py::arg("stepRule"),
        py::arg("c1"), py::arg("c2"), py::arg("alpha0"), py::arg("tol") = 0.01);

    m.def("relaxedFunction", &relaxedFunction,
        R"pbdoc(
          Applies the alphaBB relaxation method to a given function.

          Parameters
          ----------
          x : np.ndarray
              The point at which the function is evaluated. It should be a 1D array of decision variables.
          function : Callable[[np.ndarray], Union[float, np.ndarray]]
              The function to be relaxed. It takes an array as input and returns a scalar or array.
          alpha : float
              The relaxation parameter. Controls the degree of relaxation applied to the function.
          lb : np.ndarray
              The lower bounds for the decision variables. Should be the same shape as `x`.
          ub : np.ndarray
              The upper bounds for the decision variables. Should be the same shape as `x`.

          Returns
          -------
          Union[float, np.ndarray]
              The relaxed function value at point `x`. The output is of the same type as the original function's output.
      )pbdoc",
        py::arg("x"), py::arg("function"), py::arg("alpha"), py::arg("lb"), py::arg("ub"));

    m.def("computeUpperBounds", &computeUpperBounds,
        R"pbdoc(
          Compute upper bounds for a set of nodes using scipy.optimize.minimize in Python.

          This function iterates over a list of nodes, each containing lower and upper bounds
          (lb and ub) for decision variables. For each node, if the upper bound (`ubd`) is
          set to infinity, it computes the midpoint between the lower and upper bounds, then
          uses the L-BFGS-B method from `scipy.optimize.minimize` to minimize the objective
          function over the specified bounds. The computed upper bound is then stored in
          the node.

          Parameters
          ----------
          nodes : list of dict
              A list of nodes, where each node is a dictionary containing:
              - "lb": list of float
                  The lower bounds for the decision variables.
              - "ub": list of float
                  The upper bounds for the decision variables.
              - "ubd": float
                  The current upper bound. If this is set to infinity, the function
                  will attempt to compute a new upper bound.
          objective : callable
              The objective function to be minimized. This function should take a list or
              NumPy array of decision variables as input and return a scalar value.

          Returns
          -------
          list of dict
              The updated list of nodes, where each node now has a computed upper bound
              (`ubd`) if it was initially set to infinity.

          Example
          -------
          >>> import your_module_name
          >>> def objective(x):
          ...     return x[0]**2 + x[1]**2
          >>> nodes = [
          ...     {"lb": [0, 0], "ub": [2, 2], "ubd": float('inf')},
          ...     {"lb": [1, 1], "ub": [3, 3], "ubd": float('inf')}
          ... ]
          >>> updated_nodes = your_module_name.computeUpperBounds(nodes, objective)
          >>> print(updated_nodes)
          [
              {'lb': [0.0, 0.0], 'ub': [2.0, 2.0], 'ubd': <computed_value>},
              {'lb': [1.0, 1.0], 'ub': [3.0, 3.0], 'ubd': <computed_value>}
          ]

          Notes
          -----
          - This function requires that you have `scipy` installed in your Python environment.
          - The `objective` function should be differentiable for best results with the
            L-BFGS-B method.

          )pbdoc",
        py::arg("nodes"), py::arg("objective"));

}
