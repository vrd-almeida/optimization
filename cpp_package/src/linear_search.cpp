#include "funcs.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <functional>
#include <cmath>
#include <limits>


// Struct to hold the result of the Armijo line search
struct ArmijoResult {
    double alpha;  // Optimal step size
    int iterations;  // Number of iterations to find alpha
};


// Implementation of the Armijo method to find alpha
ArmijoResult armijo(
    const std::function<double(const std::vector<double>&)>& func,
    const std::vector<double>& xcur,
    const std::vector<double>& search_direction,
    double c1,
    double alpha0
) {
    // Validation
    if (c1 <= 0.0 || c1 >= 1.0) {
        throw std::invalid_argument("c1 must be a positive value less than 1.");
    }
    if (alpha0 <= 0.0) {
        throw std::invalid_argument("alpha0 must be a positive value.");
    }

    double alpha = alpha0;
    std::vector<double> xnew(xcur.size());
    double fcur = func(xcur);
    double fnew;
    std::vector<double> gradientCur = gradient(func, xcur);
    int numiter = 0;

    do {
        for (size_t i = 0; i < xcur.size(); ++i) {
            xnew[i] = xcur[i] + alpha * search_direction[i];
        }
        fnew = func(xnew);

        if (fnew <= fcur + c1 * alpha * dot_product(gradientCur, search_direction)) {
            break;
        }

        alpha /= 2.0;
        ++numiter;

    } while (true);

    return { alpha, numiter };  // Corrected: No need to cast numiter
}


// Struct to hold the result of the Wolfe line search
struct WolfeResult {
    double alpha;  // Optimal step size
    int iterations;  // Number of iterations to find alpha
};

// Implementation of the Wolfe line search
WolfeResult wolfe(
    const std::function<double(const std::vector<double>&)>& func,
    const std::vector<double>& xcur,
    const std::vector<double>& searchdirection,
    double c1,
    double c2,
    double alpha0
) {
    double alpha = alpha0;
    std::vector<double> xnew(xcur.size());
    for (size_t i = 0; i < xcur.size(); ++i) {
        xnew[i] = xcur[i] + alpha * searchdirection[i];
    }

    double fcur = func(xcur);
    double fnew = func(xnew);
    std::vector<double> gradientCur = gradient(func, xcur);
    std::vector<double> gradientNew = gradient(func, xnew);
    int numiter = 0;

    double lb = 0.0;
    double ub = std::numeric_limits<double>::infinity();

    // Check for Wolfe conditions
    while (true) {
        numiter++;
        if (fnew > fcur + c1 * alpha * dot_product(gradientCur, searchdirection)) {
            ub = alpha;
            alpha = 0.5 * (lb + ub);
        }
        else if (dot_product(gradientNew, searchdirection) < c2 * dot_product(gradientCur, searchdirection)) {
            lb = alpha;
            if (std::isinf(ub)) {
                alpha = 2.0 * lb;
            }
            else {
                alpha = 0.5 * (lb + ub);
            }
        }
        else {
            break;
        }

        // Update xnew, fnew, and gradientNew
        for (size_t i = 0; i < xcur.size(); ++i) {
            xnew[i] = xcur[i] + alpha * searchdirection[i];
        }
        fnew = func(xnew);
        gradientNew = gradient(func, xnew);
    }

    return { alpha, numiter };
}


// Steepest Descent algorithm implementation
SteepestDescentResult steepest_descent(
    const std::function<double(const std::vector<double>&)>& func,
    const std::vector<double>& x0,
    const std::string& stepRule,
    double c1,
    double c2,
    double alpha0,
    double tol
) {
    // Validation
    if (!(0 < c1 && c1 < 1)) {
        throw std::invalid_argument("c1 must be in the range (0, 1)");
    }
    if (!(0 < tol && tol < 1)) {
        throw std::invalid_argument("tol must be in the range (0, 1)");
    }

    std::vector<double> xCur = x0;
    double fcur = func(xCur);
    std::vector<double> gCur = gradient(func, xCur);

    double nmg0 = std::sqrt(dot_product(gradient(func, x0), gradient(func, x0)));

    int it = 0;
    int ls_iter = 0;

    std::vector<std::vector<double>> xIter;
    xIter.push_back(x0);

    while (std::sqrt(dot_product(gCur, gCur)) > tol * std::min(1.0, nmg0)) {
        it += 1;

        // Calculate descent direction
        std::vector<double> direction(gCur.size());
        for (size_t i = 0; i < gCur.size(); ++i) {
            direction[i] = -1.0 * gCur[i];
        }

        // Calculate step-length
        double alpha;
        int ls_ = 0;
        if (stepRule == "armijo") {
            auto result = armijo(func, xCur, direction, c1, alpha0);
            alpha = result.alpha;
            ls_ = result.iterations;
            // } else if (stepRule == "wolfe") {
            //     auto result = wolfe_line_search(function, xCur, direction, c1, c2, alpha0); // Assuming wolfe returns a similar structure
            //     alpha = result.alpha;
            //     ls_ = result.iterations;
        }
        else {
            alpha = 0.01;
        }
        ls_iter += ls_;

        // Update current point
        for (size_t i = 0; i < xCur.size(); ++i) {
            xCur[i] += alpha * direction[i];
        }
        gCur = gradient(func, xCur);
        fcur = func(xCur);
        xIter.push_back(xCur);
    }

    return { xCur, fcur, it, xIter, ls_iter };
}
