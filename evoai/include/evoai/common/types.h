#ifndef EVOAI__COMMON__TYPES_H
#define EVOAI__COMMON__TYPES_H

#include <Eigen/Dense>

#include <cstdint>

namespace evoai
{
using ValueType = double;
using IndexType = int32_t;

template <IndexType N>
using Vector = Eigen::Matrix<ValueType, N, 1>;

template <IndexType N, IndexType M>
using Matrix = Eigen::Matrix<ValueType, N, M>;

template <typename Derived>
using MatrixBase = Eigen::MatrixBase<Derived>;

}  // namespace evoai

#endif  // ifndef EVOAI__COMMON__TYPES_H
