#ifndef EVOAI__EVOLUTION__EVOLUTION_H
#define EVOAI__EVOLUTION__EVOLUTION_H

#include <evoai/common/types.h>
#include <evoai/evolution/mutation/tingri.h>
#include <evoai/evolution/population.h>
#include <evoai/graph/aggregation/accumulator.h>
#include <evoai/graph/graph.h>

#include <cassert>
#include <iostream>
#include <numeric>

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

inline Scores Fitness(Scores const& losses)
{
    Scores fitness;
    fitness.resize(losses.size());
    auto max = *std::max_element(losses.begin(), losses.end());
    std::transform(losses.begin(), losses.end(), fitness.begin(), [&max](auto loss) {
        return static_cast<ValueType>(1.0) - (loss / max);
    });
    return fitness;
}

template <typename GraphType, typename Properties>
std::tuple<Population<GraphType, Properties>, ActivationSummary<GraphType::k_total_neurons>> Select(
    Population<GraphType, Properties> const& population_in,
    ActivationSummary<GraphType::k_total_neurons> const& activations_in,
    Scores const& fitness)
{
    Population<GraphType, Properties> population_out;
    ActivationSummary<GraphType::k_total_neurons> activations_out;
    population_out.reserve(population_in.size());
    activations_out.reserve(population_in.size());

    auto const length = std::accumulate(fitness.begin(), fitness.end(), static_cast<ValueType>(0));
    auto in_index = static_cast<IndexType>(population_in.size() * 0.5);
    auto out_index = 0;
    auto const select_interval = length / static_cast<ValueType>(population_in.size());
    auto cummulative_sum_in = static_cast<ValueType>(0.0);
    auto cummulative_sum_out = select_interval;

    if (length < std::numeric_limits<ValueType>::min())
    {
        std::copy(population_in.begin(), population_in.end(), population_out.begin());
        std::copy(activations_in.begin(), activations_in.end(), activations_out.begin());
    }
    else
    {
        while (out_index < static_cast<IndexType>(population_in.size()))
        {
            if (cummulative_sum_in + fitness[in_index] >= cummulative_sum_out)
            {
                population_out.push_back(population_in[in_index]);
                population_out.back().fitness_score += fitness[in_index];
                activations_out.push_back(activations_in[in_index]);
                cummulative_sum_out += select_interval;
                out_index++;
            }
            else
            {
                cummulative_sum_in += fitness[in_index];
                in_index++;
                if (in_index >= static_cast<IndexType>(population_in.size()))
                {
                    in_index = 0;
                }
            }
        }
    }
    return {population_out, activations_out};
}

template <typename GraphType, typename MutationStrategy, typename GeneratorType>
void Mutate(Population<GraphType, typename MutationStrategy::Properties>& population,
            MutationStrategy const& strategy,
            ActivationSummary<GraphType::k_total_neurons> const& activation_summary,
            GeneratorType& generator)
{
    for (IndexType i = 0; i < static_cast<ValueType>(population.size()); ++i)
    {
        auto& specimen = population[i];
        auto const& activations = activation_summary[i];
        strategy(specimen, activations, generator);
    }
}

template <typename Loss,
          typename GraphType,
          typename MutationStrategy,
          typename InputDerived,
          typename TruthDerived,
          typename GeneratorType>
Population<GraphType, typename MutationStrategy::Properties> Evolve(
    Population<GraphType, typename MutationStrategy::Properties> const& population_in,
    MutationStrategy const& strategy,
    MatrixBase<InputDerived> const& input,
    MatrixBase<TruthDerived> const& truth,
    GeneratorType& generator)
{
    auto [losses, activations] = Score<Loss>(population_in, input, truth);
    auto fitness = Fitness(losses);
    auto [population_out, activations_out] = Select(population_in, activations, fitness);
    Mutate(population_out, strategy, activations_out, generator);
    return population_out;
}

}  // namespace detail

template <typename GraphType,
          typename Loss,
          typename MutationStrategy = mutation::Tingri,
          typename RandomGenerator = std::default_random_engine>
struct Evolution
{
    GraphType Evolve(std::vector<Vector<GraphType::k_input_neurons>> const& input,
                     std::vector<Vector<GraphType::k_output_neurons>> const& truth,
                     bool verbose = false)
    {
        assert(input.size() == truth.size());

        std::vector<IndexType> indices;
        indices.resize(input.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), generator);

        IndexType index = 0;
        if (verbose)
        {
            std::cout << "Creating Population..." << std::flush;
        }
        auto population = detail::CreatePopulation<GraphType>(population_size, strategy, generator);
        if (verbose)
        {
            std::cout << "OK" << std::endl;
        }

        for (IndexType i = 0; i < iterations; ++i)
        {
            if (verbose)
            {
                std::cout << "\r" << (i + 1) << " / " << iterations << std::flush;
            }
            detail::Evolve<Loss>(population, strategy, input[indices[index]], truth[indices[index]], generator);
            ++index;
            if (index > static_cast<IndexType>(input.size()))
            {
                index = 0;
            }
        }

        auto fittest_it =
            std::max_element(population.begin(), population.end(), [](auto const& left, auto const& right) {
                return left.fitness_score < right.fitness_score;
            });

        if (verbose)
        {
            std::cout << "\nDone" << std::endl;
        }
        return {fittest_it->adjacency};
    }

    IndexType population_size = 100;
    IndexType iterations = 1000;
    MutationStrategy strategy{};
    RandomGenerator generator{1234};
};

}  // namespace evoai

#endif  // EVOAI__EVOLUTION__EVOLUTION_H
