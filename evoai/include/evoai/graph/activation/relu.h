#ifndef EVOAI__GRAPH__ACTIVATION__RELU_H
#define EVOAI__GRAPH__ACTIVATION__RELU_H

#include <evoai/common/types.h>

namespace evoai
{
namespace activation
{

namespace detail
{
struct ActivationFunction
{
    static constexpr bool k_is_activation_function = true;
};
}  // namespace detail

struct RelU : public detail::ActivationFunction
{
    RelU() = default;
    RelU(ValueType p_k, ValueType p_d) : k{p_k}, d{p_d} {}

    template <typename Derived>
    auto operator()(MatrixBase<Derived> const& x) const
        -> decltype(((ValueType() * x).array() + ValueType()).cwiseMax(ValueType()).matrix())
    {
        return ((k * x).array() + d).cwiseMax(static_cast<ValueType>(0)).matrix();
    }

    ValueType k = 1.0;
    ValueType d = 0.0;
};
}  // namespace activation
}  // namespace evoai

#endif  // ifndef EVOAI__GRAPH__ACTIVATION__RELU_H
