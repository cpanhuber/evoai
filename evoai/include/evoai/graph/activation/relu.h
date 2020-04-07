#ifndef EVOAI__GRAPH__ACTIVATION__RELU_H
#define EVOAI__GRAPH__ACTIVATION__RELU_H

#include <evoai/common/types.h>

namespace evoai
{
namespace activation
{
struct RelU
{
    RelU() = default;
    RelU(ValueType p_k, ValueType p_d) : k{p_k}, d{p_d} {}

    template <int32_t N>
    Vector<N> operator()(Vector<N> const& x) const
    {
        return Vector<N>{((k * x).array() + d).cwiseMax(static_cast<ValueType>(0))};
    }

    ValueType k = 1.0;
    ValueType d = 0.0;
};
}  // namespace activation
}  // namespace evoai

#endif  // ifndef EVOAI__GRAPH__ACTIVATION__RELU_H
