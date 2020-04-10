#ifndef EVOAI__GRAPH__ACTIVATION__RELU_H
#define EVOAI__GRAPH__ACTIVATION__RELU_H

#include <evoai/common/types.h>
#include <evoai/graph/activation/activator.h>

namespace evoai
{
namespace activation
{

struct RelU : public detail::Activator
{
    RelU() = default;
    RelU(ValueType p_k, ValueType p_d) : k{p_k}, d{p_d} {}

    template <typename Derived>
    auto operator()(MatrixBase<Derived> const& x) const
    {
        return ((k * x).array() + d).cwiseMax(static_cast<ValueType>(0)).matrix();
    }

    ValueType k = 1.0;
    ValueType d = 0.0;
};
}  // namespace activation
}  // namespace evoai

#endif  // ifndef EVOAI__GRAPH__ACTIVATION__RELU_H
