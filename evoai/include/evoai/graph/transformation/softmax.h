#ifndef EVOAI__GRAPH__TRANSFORMATION__SOFTMAX_H
#define EVOAI__GRAPH__TRANSFORMATION__SOFTMAX_H

#include <evoai/common/types.h>

#include <cmath>
#include <utility>

namespace evoai
{

namespace transformation
{

namespace detail
{
struct Transformator
{
    static constexpr bool k_is_transformator = true;
};
}  // namespace detail

struct LogSoftmax : public detail::Transformator
{
    template <typename Derived>
    auto operator()(MatrixBase<Derived> const& in)
    {
        auto inLessMax = in.array() - in.maxCoeff();
        return (inLessMax - std::log(inLessMax.exp().sum())).matrix();
    }
};
}  // namespace transformation

}  // namespace evoai

#endif  // EVOAI__GRAPH__TRANSFORMATION__SOFTMAX_H

