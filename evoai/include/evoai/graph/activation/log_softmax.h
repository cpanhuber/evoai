#ifndef EVOAI__GRAPH__ACTIVATION__LOG_SOFTMAX_H
#define EVOAI__GRAPH__ACTIVATION__LOG_SOFTMAX_H

#include <evoai/common/types.h>
#include <evoai/graph/activation/activator.h>

#include <cmath>
#include <utility>

namespace evoai
{

namespace activation
{

struct LogSoftmax : public detail::Activator
{
    template <typename Derived>
    auto operator()(MatrixBase<Derived> const& in)
    {
        auto inLessMax = in.array() - in.maxCoeff();
        return (inLessMax - std::log(inLessMax.exp().sum())).matrix();
    }
};
}  // namespace activation

}  // namespace evoai

#endif  // EVOAI__GRAPH__ACTIVATION__LOG_SOFTMAX_H

