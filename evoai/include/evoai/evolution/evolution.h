#ifndef EVOAI__EVOLUTION__EVOLUTION_H
#define EVOAI__EVOLUTION__EVOLUTION_H

#include <evoai/common/types.h>
#include <evoai/evolution/population.h>
#include <evoai/graph/aggregation/counter.h>

namespace evoai
{

namespace detail
{
template <IndexType TotalN, typename ActivationFunctionType>
struct ActivationTracker : public ActivationFunctionType
{
    template <typename Derived>
    auto operator()(MatrixBase<Derived> const& x)
    {
        auto activation = ActivationFunctionType::operator()(x);
        counted_activations = counter(counted_activations, activation);
        return activation;
    }
    Vector<TotalN> counted_activations;
    aggregation::Counter aggregator;
};

}  // namespace detail

template <typename GraphType, typename Properties>
Population<GraphType, Properties> CreateRandomPopulation()
{
    Population<GraphType, Properties> population;
}

}  // namespace evoai

#endif  // EVOAI__EVOLUTION__EVOLUTION_H
