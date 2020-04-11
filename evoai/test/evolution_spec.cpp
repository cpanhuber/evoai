#include <evoai/evolution/population.h>

#include <evoai/common/types.h>
#include <evoai/graph/activation/relu.h>
#include <evoai/graph/aggregation/aggregator.h>

#include "mock/mock_random_generator.h"

#include <gtest/gtest.h>

#include <random>

namespace
{

using namespace evoai;

TEST(Mutation, CreateWeightMutationFactors_WhenMutancyOne)
{
    Vector<6> activation_counts;
    activation_counts << 1.0, 2.0, 6.0, 0.0, 2.0, 1.0;

    auto result = detail::CreateWeightMutationFactors(activation_counts, 1.0);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(0.5), result(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(3.0), result(2), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.0), result(3), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(4), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.5), result(5), tolerance);
}

TEST(Mutation, CreateWeightMutationFactors_WhenEqual)
{
    Vector<6> activation_counts;
    activation_counts << 3.0, 3.0, 3.0, 3.0, 3.0, 3.0;

    auto result = detail::CreateWeightMutationFactors(activation_counts, 1.0);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(2), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(3), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(4), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(5), tolerance);
}

TEST(Mutation, CreateWeightMutationFactors_WhenTypical)
{
    Vector<6> activation_counts;
    activation_counts << 1.0, 2.0, 6.0, 0.0, 2.0, 1.0;

    auto result = detail::CreateWeightMutationFactors(activation_counts, 0.1);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(0.95), result(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.2), result(2), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.9), result(3), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(4), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.95), result(5), tolerance);
}

// This is highly likely, but not guaranteed to work
TEST(Mutation, DISABLED_SampleNormal)
{
    std::default_random_engine generator;
    ValueType samples = 0;
    for (int i = 0; i < 1000; ++i)
    {
        samples += detail::SampleNormal(5.0, 2.0, generator);
    }

    auto tolerance = 1e-1;
    EXPECT_NEAR(5.0, samples / 1000.0, tolerance);
}

TEST(Mutation, MutateAdjacency_WhenAdjacencyOneNoMutation)
{
    Matrix<6, 6> adjacency;
    adjacency.fill(1.0);
    Vector<6> weight_mutation_factors;
    weight_mutation_factors << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

    detail::MutateAdjacency(adjacency, weight_mutation_factors);

    Matrix<6, 6> expected = Matrix<6, 6>::Ones();
    EXPECT_TRUE(adjacency == expected);
}

TEST(Mutation, MutateAdjacency_WhenAdjacencyOneTypicalMutation)
{
    Matrix<6, 6> adjacency = Matrix<6, 6>::Ones();
    Vector<6> weight_mutation_factors;
    weight_mutation_factors << 0.95, 1.2, 0.9, 1.1, 0.9, 1.3;

    detail::MutateAdjacency(adjacency, weight_mutation_factors);

    Matrix<6, 6> expected = Vector<6>::Ones() * weight_mutation_factors.transpose();
    EXPECT_TRUE(adjacency == expected);
}

TEST(Mutation, MutateAdjacency_WhenTypical)
{
    Matrix<2, 2> adjacency;
    adjacency << 0.7, -1.2, 0.3, -0.3;
    Vector<2> weight_mutation_factors;
    weight_mutation_factors << 0.95, 1.2;

    detail::MutateAdjacency(adjacency, weight_mutation_factors);

    Matrix<2, 2> expected;
    expected << 0.7 * 0.95, -1.2 * 1.2, 0.3 * 0.95, -0.3 * 1.2;
    EXPECT_TRUE(adjacency == expected);
}

}  // namespace
