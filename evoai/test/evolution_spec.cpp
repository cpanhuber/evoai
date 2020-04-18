#include <evoai/evolution/evolution.h>
#include <evoai/evolution/mutation/tingri.h>
#include <evoai/evolution/population.h>

#include <evoai/common/types.h>
#include <evoai/graph/activation/relu.h>
#include <evoai/graph/graph.h>
#include <evoai/graph/loss/mean_squared_error.h>

#include "mock/mock_random_generator.h"

#include <gtest/gtest.h>

namespace
{

using namespace evoai;

TEST(Population, CreatePopulation)
{
    mutation::Tingri tingri;
    tingri.settings.kill_initial_mean = 1.0;
    tingri.settings.kill_change_deviation = 0.0;
    tingri.settings.revive_initial_mean = 2.0;
    tingri.settings.revive_change_deviation = 0.0;
    tingri.settings.mutancy_initial_mean = 3.0;
    tingri.settings.mutancy_change_deviation = 0.0;
    tingri.settings.weight_initial_mean = 4.0;
    tingri.settings.weight_initial_deviation = 0.0;

    test::MockRandomGenerator mock_generator;
    mock_generator.output = 1000000000;  // ~0.24 in uniform([0, 1]) for gcc implementation

    using GraphType = NeuralGraph<2, 1, 3, 5, activation::RelU, aggregation::Accumulator, activation::RelU>;
    auto population = detail::CreatePopulation<GraphType>(100, tingri, mock_generator);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_EQ(100, population.size());
    std::for_each(population.begin(), population.end(), [&tolerance](auto& specimen) {
        EXPECT_NEAR(static_cast<ValueType>(3.0), specimen.mutancy, tolerance);
        EXPECT_NEAR(static_cast<ValueType>(2.0), specimen.revive_prior, tolerance);
        EXPECT_NEAR(static_cast<ValueType>(1.0), specimen.kill_prior, tolerance);
        EXPECT_TRUE((specimen.adjacency.array() == static_cast<ValueType>(4.0)).all());
    });
}

class EvolutionFixture : public ::testing::Test
{
  public:
    using ActivationTracker = detail::ActivationTracker<4, activation::RelU>;
    using GraphType = NeuralGraph<2, 1, 1, 3, ActivationTracker, aggregation::Accumulator, activation::RelU>;
    void SetUp()
    {
        adjacency(2, 0) = 1.0;
        adjacency(2, 1) = 1.5;
        adjacency(3, 2) = 2.0;
    }
    GraphType::AdjacencyType adjacency = GraphType::Traits::AdjacencyType::Zero();
    ActivationTracker tracker;
    Vector<GraphType::k_input_neurons> input;
};

TEST_F(EvolutionFixture, ActivationTracker_WhenNoActivations)
{
    input << -1.0, -2.0;
    detail::Predict<GraphType>(input, adjacency, tracker);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(0.0), tracker.accumulated_activations(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.0), tracker.accumulated_activations(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.0), tracker.accumulated_activations(2), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.0), tracker.accumulated_activations(3), tolerance);
}

TEST_F(EvolutionFixture, ActivationTracker_WhenSomeActivations)
{
    input << 1.0, 0.0;
    detail::Predict<GraphType>(input, adjacency, tracker);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(3.0), tracker.accumulated_activations(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.0), tracker.accumulated_activations(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(2.0), tracker.accumulated_activations(2), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(2.0), tracker.accumulated_activations(3), tolerance);
}

TEST_F(EvolutionFixture, ActivationTracker_WhenAllActivations)
{
    input << 1.0, 2.0;
    detail::Predict<GraphType>(input, adjacency, tracker);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(3.0), tracker.accumulated_activations(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(6.0), tracker.accumulated_activations(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(8.0), tracker.accumulated_activations(2), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(8.0), tracker.accumulated_activations(3), tolerance);
}

TEST_F(EvolutionFixture, ActivationTracker_WhenRecursiveActivations)
{
    adjacency(2, 2) = 1.0;

    input << 1.0, 2.0;
    detail::Predict<GraphType>(input, adjacency, tracker);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(3.0), tracker.accumulated_activations(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(6.0), tracker.accumulated_activations(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(12.0), tracker.accumulated_activations(2), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(8.0), tracker.accumulated_activations(3), tolerance);
}

TEST_F(EvolutionFixture, Score)
{
    using Population = detail::Population<GraphType, mutation::Tingri::Properties>;
    using Specimen = detail::Specimen<GraphType, mutation::Tingri::Properties>;
    Specimen specimen;
    specimen.adjacency = adjacency;
    Population population;
    population.push_back(specimen);
    input << 1.0, 2.0;
    Vector<1> truth;
    truth << 14.0;

    // graph output:
    // 1.0 1.0 1.0 1.0
    // 2.0 2.0 2.0 2.0
    // 0.0 4.0 4.0 4.0
    // 0.0 0.0 8.0 8.0 = 16.0

    auto [scores, activations] = detail::Score<loss::MeanSquaredError>(population, input, truth);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(4.0), scores[0], tolerance);
    EXPECT_NEAR(static_cast<ValueType>(3.0), activations[0](0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(6.0), activations[0](1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(8.0), activations[0](2), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(8.0), activations[0](3), tolerance);
}

TEST(Evolution, Fitness)
{
    detail::Scores losses{0.0, 3.0, 1.5, 6.0};

    auto fitness = detail::Fitness(losses);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(1.0), fitness[0], tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.5), fitness[1], tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.75), fitness[2], tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.0), fitness[3], tolerance);
}

TEST(Evolution, Select)
{
    using ActivationTracker = detail::ActivationTracker<4, activation::RelU>;
    using GraphType = NeuralGraph<2, 1, 1, 3, ActivationTracker, aggregation::Accumulator, activation::RelU>;
    detail::Scores fitness{0.3, 0.4, 0.4, 0.8, 0.1};
    detail::Population<GraphType, mutation::Tingri::Properties> population_in;
    population_in.resize(5);
    population_in[0].mutancy = static_cast<ValueType>(0.0);
    population_in[1].mutancy = static_cast<ValueType>(1.0);
    population_in[2].mutancy = static_cast<ValueType>(2.0);
    population_in[3].mutancy = static_cast<ValueType>(3.0);
    population_in[4].mutancy = static_cast<ValueType>(4.0);
    detail::ActivationSummary<4> activation_summary;
    activation_summary.resize(5);
    activation_summary[0].fill(0.0);
    activation_summary[1].fill(1.0);
    activation_summary[2].fill(2.0);
    activation_summary[3].fill(3.0);
    activation_summary[4].fill(4.0);

    auto [population_out, activations_out] = Select(population_in, activation_summary, fitness);

    // 2, 3, 3, 0, 1
    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(2.0), population_out[0].mutancy, tolerance);
    EXPECT_NEAR(static_cast<ValueType>(3.0), population_out[1].mutancy, tolerance);
    EXPECT_NEAR(static_cast<ValueType>(3.0), population_out[2].mutancy, tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.0), population_out[3].mutancy, tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), population_out[4].mutancy, tolerance);

    EXPECT_NEAR(static_cast<ValueType>(2.0), activations_out[0](0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(3.0), activations_out[1](0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(3.0), activations_out[2](0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.0), activations_out[3](0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), activations_out[4](0), tolerance);
}

TEST(Evolution, Mutate)
{
    test::MockRandomGenerator mock_generator;
    mock_generator.output = 1000000000;  // ~0.24 in uniform([0, 1]) for gcc implementation

    using GraphType = NeuralGraph<2, 1, 3, 5, activation::RelU, aggregation::Accumulator, activation::RelU>;
    detail::Specimen<GraphType, mutation::Tingri::Properties> specimen;
    specimen.mutancy = 0.5;
    specimen.kill_prior = 0.5;
    specimen.revive_prior = 0.5;
    Vector<6> activations;
    activations << 4.0, 1.0, 0.0, 1.0, 7.0, 5.0;
    detail::Population<GraphType, mutation::Tingri::Properties> population;
    population.push_back(specimen);
    mutation::Tingri tingri;
    detail::ActivationSummary<6> activation_summary;
    activation_summary.push_back(activations);

    detail::Mutate(population, tingri, activation_summary, mock_generator);

    auto expected = mutation::detail::SampleNormal(0.5, 0.01, mock_generator);
    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(expected, population[0].mutancy, tolerance);
    EXPECT_NEAR(expected, population[0].revive_prior, tolerance);
    EXPECT_NEAR(expected, population[0].kill_prior, tolerance);
}

}  // namespace
