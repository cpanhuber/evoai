#ifndef EVOAI__EVOLUTION__EVOLUTION_H
#define EVOAI__EVOLUTION__EVOLUTION_H

#include <evoai/common/types.h>
#include <evoai/evolution/population.h>
#include <evoai/graph/aggregation/accumulator.h>
#include <evoai/graph/graph.h>

namespace evoai
{

namespace detail
{
template <IndexType TotalN, typename ActivationFunctionType>
struct ActivationTracker : public ActivationFunctionType
{
    ActivationTracker() : aggregator{}, accumulated_activations{aggregation::CreateInitial<TotalN>(aggregator)} {}
    template <typename Derived>
    auto operator()(MatrixBase<Derived> const& x)
    {
        auto activation = ActivationFunctionType::operator()(x);
        accumulated_activations = aggregator(accumulated_activations, activation);
        return activation;
    }
    aggregation::Accumulator aggregator;
    Vector<TotalN> accumulated_activations;
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
