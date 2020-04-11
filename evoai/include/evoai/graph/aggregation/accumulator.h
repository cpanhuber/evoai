#ifndef EVOAI__GRAPH__AGGREGATION__ACCUMULTATOR_H
#define EVOAI__GRAPH__AGGREGATION__ACCUMULTATOR_H

#include <evoai/common/types.h>
#include <evoai/graph/aggregation/aggregator.h>

#include <climits>

namespace evoai
{

namespace aggregation
{

struct Accumulator : public detail::Aggregator
{
    template <typename DerivedLeft, typename DerivedRight>
    auto operator()(MatrixBase<DerivedLeft> const& left, MatrixBase<DerivedRight> const& right)
    {
        return left + right;
    }
};

template <IndexType N>
Vector<N> CreateInitial(Accumulator const&)
{
    return Vector<N>::Zero();
}

}  // namespace aggregation
}  // namespace evoai

#endif  // ifndef EVOAI__GRAPH__AGGREGATION__ACCUMULATOR_H
