#ifndef EVOAI__GRAPH__LOSS__LOSS_FUNCTION_H
#define EVOAI__GRAPH__LOSS__LOSS_FUNCTION_H

#include <evoai/common/types.h>

namespace evoai
{
namespace loss
{

namespace detail
{
struct LossFunction
{
    static constexpr bool k_is_loss_function = true;
};
}  // namespace detail
}  // namespace loss
}  // namespace evoai

#endif  // ifndef EVOAI__GRAPH__LOSS__LOSS_FUNCTION_H
