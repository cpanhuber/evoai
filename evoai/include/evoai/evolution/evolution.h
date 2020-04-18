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

template <IndexType N>
using ActivationSummary = std::vector<Vector<N>>;

using Scores = std::vector<ValueType>;

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

template <typename Loss, typename Properties, typename GraphType, typename InputDerived, typename TruthDerived>
std::tuple<Scores, ActivationSummary<GraphType::k_total_neurons>> Score(
    Population<GraphType, Properties> const population_in,
    MatrixBase<InputDerived> const& input,
    MatrixBase<TruthDerived> const& truth)
{
    Scores scores;
    scores.reserve(population_in.size());
    ActivationSummary<GraphType::k_total_neurons> activations;
    activations.reserve(population_in.size());

    std::for_each(population_in.begin(), population_in.end(), [&truth, &input, &scores, &activations](auto& specimen) {
        Loss loss;
        using ActivationTrackerType =
            ActivationTracker<GraphType::k_total_neurons, typename GraphType::ActivationFunctionType>;
        using Traits = GraphTraits<GraphType::k_input_neurons,
                                   GraphType::k_output_neurons,
                                   GraphType::k_hidden_neurons,
                                   GraphType::k_forward_iterations,
                                   ActivationTrackerType,
                                   typename GraphType::AggregatorType,
                                   typename GraphType::TransformatorType>;

        ActivationTrackerType tracker;
        auto prediction = detail::Predict<Traits>(input, specimen.adjacency, tracker);
        scores.push_back(loss(prediction, truth));
        activations.push_back(tracker.accumulated_activations);
    });

    return {scores, activations};
}

}  // namespace detail

}  // namespace evoai

#endif  // EVOAI__EVOLUTION__EVOLUTION_H
