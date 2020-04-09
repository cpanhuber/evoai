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

struct Accumulator : public detail::Aggregator
{
    template <typename DerivedLeft, typename DerivedRight>
    auto operator()(MatrixBase<DerivedLeft> const& left, MatrixBase<DerivedRight> const& right)
    {
        return left + right;
    }
};  // namespace aggregation

template <IndexType N>
Vector<N> CreateInitial(Accumulator const&)
{
    return Vector<N>::Zero();
}

}  // namespace aggregation
}  // namespace evoai

#endif  // ifndef EVOAI__GRAPH__AGGREGATION__AGGREGATOR_H
