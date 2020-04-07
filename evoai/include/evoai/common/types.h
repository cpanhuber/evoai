#ifndef EVOAI__COMMON__TYPES_H
#define EVOAI__COMMON__TYPES_H

#include <Eigen/Dense>

#include <cstddef>

namespace evoai
{
using ValueType = double;

template <size_t N>
using Vector = Eigen::Matrix<ValueType, N, 1>;

template <size_t N, size_t M>
using Matrix = Eigen::Matrix<ValueType, N, M>;

}  // namespace evoai

#endif  // ifndef EVOAI__COMMON__TYPES_H
