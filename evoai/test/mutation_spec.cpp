#include <evoai/evolution/evolution.h>
#include <evoai/evolution/mutation/tingri.h>

#include <evoai/common/types.h>
#include <evoai/graph/activation/relu.h>
#include <evoai/graph/aggregation/aggregator.h>
#include <evoai/graph/graph.h>

#include "mock/mock_random_generator.h"

#include <gtest/gtest.h>

#include <random>

namespace
{

using namespace evoai;

TEST(Tingri, CreateWeightMutationFactors_WhenMutancyOne)
{
    Vector<6> activation_counts;
    activation_counts << 1.0, 2.0, 6.0, 0.0, 2.0, 1.0;

    auto result = mutation::detail::CreateWeightMutationFactors(activation_counts, 1.0);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(0.5), result(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(3.0), result(2), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.0), result(3), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(4), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.5), result(5), tolerance);
}

TEST(Tingri, CreateWeightMutationFactors_WhenEqual)
{
    Vector<6> activation_counts;
    activation_counts << 3.0, 3.0, 3.0, 3.0, 3.0, 3.0;

    auto result = mutation::detail::CreateWeightMutationFactors(activation_counts, 1.0);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(2), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(3), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(4), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(5), tolerance);
}

TEST(Tingri, CreateWeightMutationFactors_WhenTypical)
{
    Vector<6> activation_counts;
    activation_counts << 1.0, 2.0, 6.0, 0.0, 2.0, 1.0;

    auto result = mutation::detail::CreateWeightMutationFactors(activation_counts, 0.1);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(0.95), result(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.2), result(2), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.9), result(3), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(4), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.95), result(5), tolerance);
}

// This is highly likely, but not guaranteed to work
TEST(Tingri, DISABLED_SampleNormal)
{
    std::default_random_engine generator;
    ValueType samples = 0;
    for (int i = 0; i < 1000; ++i)
    {
        samples += mutation::detail::SampleNormal(5.0, 2.0, generator);
    }

    auto tolerance = 1e-1;
    EXPECT_NEAR(5.0, samples / 1000.0, tolerance);
}

TEST(Tingri, MutateAdjacency_WhenAdjacencyOneNoMutation)
{
    Matrix<6, 6> adjacency;
    adjacency.fill(1.0);
    Vector<6> weight_mutation_factors;
    weight_mutation_factors << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

    mutation::detail::MutateAdjacency(adjacency, weight_mutation_factors);

    Matrix<6, 6> expected = Matrix<6, 6>::Ones();
    EXPECT_TRUE(adjacency == expected);
}

TEST(Tingri, MutateAdjacency_WhenAdjacencyOneTypicalMutation)
{
    Matrix<6, 6> adjacency = Matrix<6, 6>::Ones();
    Vector<6> weight_mutation_factors;
    weight_mutation_factors << 0.95, 1.2, 0.9, 1.1, 0.9, 1.3;

    mutation::detail::MutateAdjacency(adjacency, weight_mutation_factors);

    Matrix<6, 6> expected = Vector<6>::Ones() * weight_mutation_factors.transpose();
    EXPECT_TRUE(adjacency == expected);
}

TEST(Tingri, MutateAdjacency_WhenTypical)
{
    Matrix<2, 2> adjacency;
    adjacency << 0.7, -1.2, 0.3, -0.3;
    Vector<2> weight_mutation_factors;
    weight_mutation_factors << 0.95, 1.2;

    mutation::detail::MutateAdjacency(adjacency, weight_mutation_factors);

    Matrix<2, 2> expected;
    expected << 0.7 * 0.95, -1.2 * 1.2, 0.3 * 0.95, -0.3 * 1.2;
    EXPECT_TRUE(adjacency == expected);
}

TEST(Tingri, NormalCDF)
{
    Vector<3> v;
    v << 2.1, -1.2, 0.0;

    auto result = mutation::detail::NormalCDF(v);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(0.98213558, result(0), tolerance);
    EXPECT_NEAR(0.11506967, result(1), tolerance);
    EXPECT_NEAR(0.5, result(2), tolerance);
}

TEST(Tingri, CreateKillChances)
{
    Vector<2> v;
    v << 0.1, 0.99;
    auto ones = Vector<2>::Ones();
    auto kill_prior = static_cast<ValueType>(0.5);

    auto result = mutation::detail::CreateKillChances(v, ones, kill_prior);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(0.45), result(0, 0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.45), result(1, 0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.005), result(0, 1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.005), result(1, 1), tolerance);
}

TEST(Tingri, CreateReviveChances)
{
    Vector<2> v;
    v << 0.1, 0.99;
    auto ones = Vector<2>::Ones();
    auto kill_prior = static_cast<ValueType>(0.5);

    auto result = mutation::detail::CreateReviveChances(v, ones, kill_prior);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(0.05), result(0, 0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.05), result(1, 0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.495), result(0, 1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.495), result(1, 1), tolerance);
}

TEST(Tingri, CreateKillMatrix)
{
    test::MockRandomGenerator mock_generator;
    mock_generator.output = 2000000000;  // ~0.5 in uniform([0, 1]) for gcc implementation
    std::uniform_real_distribution<ValueType> uniform{static_cast<ValueType>(0.0), static_cast<ValueType>(1.0)};
    auto treshhold = uniform(mock_generator);
    Matrix<2, 2> kill_chances;
    kill_chances << 0.1, 0.9, 0.1, 0.9;

    auto result = mutation::detail::CreateKillMatrix(kill_chances, mock_generator);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(0.1 < treshhold ? static_cast<ValueType>(1.0) : static_cast<ValueType>(0.0), result(0, 0), tolerance);
    EXPECT_NEAR(0.1 < treshhold ? static_cast<ValueType>(1.0) : static_cast<ValueType>(0.0), result(1, 0), tolerance);
    EXPECT_NEAR(0.9 < treshhold ? static_cast<ValueType>(1.0) : static_cast<ValueType>(0.0), result(0, 1), tolerance);
    EXPECT_NEAR(0.9 < treshhold ? static_cast<ValueType>(1.0) : static_cast<ValueType>(0.0), result(1, 1), tolerance);
}

TEST(Tingri, CreateReviveMatrix)
{
    test::MockRandomGenerator mock_generator;
    mock_generator.output = 2000000000;  // ~0.5 in uniform([0, 1]) for gcc implementation
    std::uniform_real_distribution<ValueType> uniform{static_cast<ValueType>(0.0), static_cast<ValueType>(1.0)};
    std::normal_distribution<ValueType> normal{static_cast<ValueType>(0.0), static_cast<ValueType>(0.5)};
    auto treshhold = uniform(mock_generator);
    auto new_weight = normal(mock_generator);
    Matrix<2, 2> revive_chances;
    revive_chances << 0.1, 0.9, 0.1, 0.9;
    Matrix<2, 2> adjacency;
    adjacency << 0.0, 1.3, 1.1, 0.0;

    auto result = mutation::detail::CreateReviveMatrix(revive_chances, adjacency, mock_generator);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(0.1 < treshhold || std::abs(adjacency(0, 0)) > 1e-5 ? static_cast<ValueType>(0.0)
                                                                    : static_cast<ValueType>(new_weight),
                result(0, 0),
                tolerance);
    EXPECT_NEAR(0.1 < treshhold || std::abs(adjacency(1, 0)) > 1e-5 ? static_cast<ValueType>(0.0)
                                                                    : static_cast<ValueType>(new_weight),
                result(1, 0),
                tolerance);
    EXPECT_NEAR(0.9 < treshhold || std::abs(adjacency(0, 1)) > 1e-5 ? static_cast<ValueType>(0.0)
                                                                    : static_cast<ValueType>(new_weight),
                result(0, 1),
                tolerance);
    EXPECT_NEAR(0.9 < treshhold || std::abs(adjacency(1, 1)) > 1e-5 ? static_cast<ValueType>(0.0)
                                                                    : static_cast<ValueType>(new_weight),
                result(1, 1),
                tolerance);
}

TEST(Tingri, ReviveOrKill)
{
    test::MockRandomGenerator mock_generator;
    mock_generator.output = 1000000000;  // ~0.24 in uniform([0, 1]) for gcc implementation

    Vector<2> deviations;
    deviations << 4.9, -5.7;
    Matrix<2, 2> adjacency;
    adjacency << 0.0, 1.3, 1.1, 0.0;
    auto revive_prior = 0.5;
    auto kill_prior = 0.5;

    mutation::detail::ReviveOrKill(adjacency, deviations, revive_prior, kill_prior, mock_generator);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_TRUE(std::abs(adjacency(0.0)) > tolerance);  // connection came alive
    EXPECT_NEAR(1.1, adjacency(1, 0), tolerance);       // connection stays alive
    EXPECT_NEAR(0.0, adjacency(0, 1), tolerance);       // connection is killed
    EXPECT_NEAR(0.0, adjacency(1, 1), tolerance);       // connection stays dead
}

TEST(Tingri, CreateStandardDeviations)
{
    Vector<4> activations;
    activations << -4.0, -1.0, 0.0, 1.0;

    auto deviations = mutation::detail::CreateStandardDeviations(activations);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(-2.0), deviations(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.0), deviations(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(2.0 / 3.0), deviations(2), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(4.0 / 3.0), deviations(3), tolerance);
}

TEST(Tingri, call_WhenTypical)
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

    mutation::Tingri tingri;
    tingri(specimen, activations, mock_generator);

    EXPECT_NEAR(static_cast<ValueType>(0.5), specimen.mutancy, static_cast<ValueType>(0.3));
    EXPECT_NEAR(static_cast<ValueType>(0.5), specimen.revive_prior, static_cast<ValueType>(0.3));
    EXPECT_NEAR(static_cast<ValueType>(0.5), specimen.kill_prior, static_cast<ValueType>(0.3));
}

TEST(Tingri, CreateSpecimen)
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
    auto specimen = tingri.CreateSpecimen<GraphType>(mock_generator);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(3.0), specimen.mutancy, tolerance);
    EXPECT_NEAR(static_cast<ValueType>(2.0), specimen.revive_prior, tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), specimen.kill_prior, tolerance);
    EXPECT_TRUE((specimen.adjacency.array() == static_cast<ValueType>(4.0)).all());
}

}  // namespace
