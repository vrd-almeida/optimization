#include <cmath>
#include "funcs.h"


// Function to calculate the Gradient vector
std::vector<double> gradient(
    const std::function<double(const std::vector<double>&)>& func,
    const std::vector<double>& x
) {
    
    int n = static_cast<int>(x.size());
    std::vector<double> grad(n);
    double value;
    
    for (int i = 0; i < n; ++i) {
        double h = adaptive_step(x[i]);
        std::vector<double> x_plus = x, x_minus = x;
        x_plus[i] += h;
        x_minus[i] -= h;

        value = (func(x_plus) - func(x_minus)) / (2 * h);
        grad[i] = round_to(value, 4);
    }
    
    return grad;
}


// Function to calculate the Hessian matrix
std::vector<std::vector<double>> hessian(
    const std::function<double(const std::vector<double>&)>& func,
    const std::vector<double>& x
) {
    
    int n = static_cast<int>(x.size());
    std::vector<std::vector<double>> hess(n, std::vector<double>(n));
    double value;
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double h_i = adaptive_step(x[i]);
            double h_j = adaptive_step(x[j]);

            std::vector<double> x_plus_plus = x, x_plus_minus = x;
            std::vector<double> x_minus_plus = x, x_minus_minus = x;

            x_plus_plus[i] += h_i; x_plus_plus[j] += h_j;
            x_plus_minus[i] += h_i; x_plus_minus[j] -= h_j;
            x_minus_plus[i] -= h_i; x_minus_plus[j] += h_j;
            x_minus_minus[i] -= h_i; x_minus_minus[j] -= h_j;

            value = (func(x_plus_plus) - func(x_plus_minus) - func(x_minus_plus) + func(x_minus_minus)) / (4 * h_i * h_j);
            hess[i][j] = round_to(value, 4);
        }
    }
    return hess;
}
