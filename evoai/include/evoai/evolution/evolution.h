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

template <typename GraphType, typename MutationStrategy, typename GeneratorType>
Population<GraphType, typename MutationStrategy::Properties> CreatePopulation(IndexType const n,
                                                                              MutationStrategy const& strategy,
                                                                              GeneratorType& generator)
{
    using PopulationType = Population<GraphType, typename MutationStrategy::Properties>;
    PopulationType population{};
    population.resize(n);

    std::for_each(population.begin(), population.end(), [&generator, &strategy](auto& specimen) {
        specimen = strategy.template CreateSpecimen<GraphType>(generator);
    });

    return population;
}

}  // namespace detail

}  // namespace evoai

#endif  // EVOAI__EVOLUTION__EVOLUTION_H
