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
    typename Traits::AdjacencyType adjacency;
};
}  // namespace detail
template <typename GraphType, typename Properties>
using Population = std::vector<detail::Specimen<GraphType, Properties>>;

}  // namespace evoai

#endif  // EVOAI__EVOLUTION__POPULATION_H
