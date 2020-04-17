#include <evoai/evolution/evolution.h>
#include <evoai/evolution/mutation/tingri.h>

#include <evoai/common/types.h>
#include <evoai/graph/activation/relu.h>
#include <evoai/graph/graph.h>

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

}  // namespace
