#include <cmath>
#include "funcs.h"


double round_to(double value, int decimal_places) {
    double scale = std::pow(10.0, decimal_places);
    return std::round(value * scale) / scale;
}


double adaptive_step(double x, double epsilon) {
    return std::sqrt(epsilon) * (std::abs(x) + 1e-8);
}


double dot_product(const std::vector<double>& v1, const std::vector<double>& v2) {
    double dot = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot += v1[i] * v2[i];
    }
    return dot;
}