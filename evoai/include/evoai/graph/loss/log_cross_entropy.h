#ifndef EVOAI__GRAPH__LOSS__LOG_CROSS_ENTROPY_H
#define EVOAI__GRAPH__LOSS__LOG_CROSS_ENTROPY_H

#include <evoai/common/types.h>
#include <evoai/graph/loss/loss_function.h>

namespace evoai
{
namespace loss
{

struct LogCrossEntropy : public detail::LossFunction
{
    template <typename DerivedPredicted, typename DerivedActual>
    auto operator()(MatrixBase<DerivedPredicted> const& y_predicted, MatrixBase<DerivedActual> const& y_actual) const
    {
        return -(y_actual.array() * y_predicted.array()).sum();
    }
};
}  // namespace loss
}  // namespace evoai

#endif  // ifndef EVOAI__GRAPH__LOSS__LOG_CROSS_ENTROPY_H
