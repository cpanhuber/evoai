#ifndef EVOAI__GRAPH__LOSS__MEAN_SQUARED_ERROR_H
#define EVOAI__GRAPH__LOSS__MEAN_SQUARED_ERROR_H

#include <evoai/common/types.h>
#include <evoai/graph/loss/loss_function.h>

namespace evoai
{
namespace loss
{

struct MeanSquaredError : public detail::LossFunction
{
    template <typename DerivedPredicted, typename DerivedActual>
    auto operator()(MatrixBase<DerivedPredicted> const& y_predicted, MatrixBase<DerivedActual> const& y_actual) const
    {
        return static_cast<ValueType>((y_predicted - y_actual).squaredNorm()) /
               static_cast<ValueType>(y_predicted.rows());
    }
};
}  // namespace loss
}  // namespace evoai

#endif  // ifndef EVOAI__GRAPH__LOSS__MEAN_SQUARED_ERROR_H
