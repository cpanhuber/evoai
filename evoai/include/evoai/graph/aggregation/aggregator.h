#ifndef EVOAI__GRAPH__AGGREGATION__AGGREGATOR_H
#define EVOAI__GRAPH__AGGREGATION__AGGREGATOR_H

#include <evoai/common/types.h>

#include <climits>

namespace evoai
{

namespace aggregation
{

namespace detail
{
struct Aggregator
{
    static constexpr bool k_is_aggregator = true;
};
}  // namespace detail
}  // namespace aggregation
}  // namespace evoai

#endif  // ifndef EVOAI__GRAPH__AGGREGATION__AGGREGATOR_H
