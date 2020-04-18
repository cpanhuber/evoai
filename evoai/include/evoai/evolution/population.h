#ifndef EVOAI__EVOLUTION__POPULATION_H
#define EVOAI__EVOLUTION__POPULATION_H

#include <evoai/common/types.h>

#include <vector>

namespace evoai
{

namespace detail
{
template <typename GraphType, typename Properties>
struct Specimen : Properties
{
    using Traits = typename GraphType::Traits;

    ValueType mutancy;
    ValueType fitness_score = static_cast<ValueType>(0);
    typename Traits::AdjacencyType adjacency;
};
template <typename GraphType, typename Properties>
using Population = std::vector<detail::Specimen<GraphType, Properties>>;

}  // namespace detail

}  // namespace evoai

#endif  // EVOAI__EVOLUTION__POPULATION_H
