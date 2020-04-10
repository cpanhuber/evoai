#ifndef EVOAI__GRAPH__ACTIVATION__ACTIVATOR_H
#define EVOAI__GRAPH__ACTIVATION__ACTIVATOR_H

#include <evoai/common/types.h>

namespace evoai
{
namespace activation
{

namespace detail
{
struct Activator
{
    static constexpr bool k_is_activation_function = true;
};
}  // namespace detail
}  // namespace activation
}  // namespace evoai

#endif  // ifndef EVOAI__GRAPH__ACTIVATION__ACTIVATOR_H
