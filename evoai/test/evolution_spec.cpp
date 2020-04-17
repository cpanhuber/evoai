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

}  // namespace
