#ifndef EVOAI__GRAPH__AGGREGATION__COUNTER_H
#define EVOAI__GRAPH__AGGREGATION__COUNTER_H

#include <evoai/common/types.h>
#include <evoai/graph/aggregation/aggregator.h>

#include <climits>

namespace evoai
{

namespace aggregation
{

struct Counter : public detail::Aggregator
{
    template <typename DerivedLeft, typename DerivedRight>
    auto operator()(MatrixBase<DerivedLeft> const& left, MatrixBase<DerivedRight> const& right)
    {
        return left + (right.array() > std::numeric_limits<ValueType>::min()).matrix().template cast<ValueType>();
    }
};

template <IndexType N>
Vector<N> CreateInitial(Counter const&)
{
    return Vector<N>::Zero();
}

}  // namespace aggregation
}  // namespace evoai

#endif  // ifndef EVOAI__GRAPH__AGGREGATION__COUNTER_H
