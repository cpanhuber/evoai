#include <evoai/graph/graph.h>

#include <evoai/common/types.h>
#include <evoai/graph/activation/relu.h>
#include <evoai/graph/aggregation/aggregator.h>

#include <gtest/gtest.h>

namespace
{

using namespace evoai;

class GraphFixture : public ::testing::Test
{
  public:
    using GraphType = NeuralGraph<2, 1, 3, 5, activation::RelU, aggregation::Accumulator, activation::RelU>;
    void SetUp() override
    {
        // Test Graph:
        // I1 X  H1 -
        // I2 X  H2 - O
        //    X  H3 -

        adjacency_matrix_ = Matrix<6, 6>::Zero();
        adjacency_matrix_(2, 0) = static_cast<ValueType>(0.2);   // I1 - H1
        adjacency_matrix_(3, 0) = static_cast<ValueType>(-0.4);  // I1 - H2
        adjacency_matrix_(4, 0) = static_cast<ValueType>(-1.2);  // I1 - H3
        adjacency_matrix_(2, 1) = static_cast<ValueType>(-0.5);  // I2 - H1
        adjacency_matrix_(3, 1) = static_cast<ValueType>(0.9);   // I2 - H2
        adjacency_matrix_(4, 1) = static_cast<ValueType>(0.8);   // I2 - H3
        adjacency_matrix_(5, 2) = static_cast<ValueType>(0.2);   // H1 - O
        adjacency_matrix_(5, 3) = static_cast<ValueType>(-0.2);  // H2 - O
        adjacency_matrix_(5, 4) = static_cast<ValueType>(0.8);   // H3 - O
        graph_ = GraphType(adjacency_matrix_);
    }

  protected:
    Matrix<6, 6> adjacency_matrix_;
    GraphType graph_;
};

TEST_F(GraphFixture, SinglePass)
{
    Vector<6> intermediate;
    intermediate << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
    activation::RelU relu;

    auto result = detail::SinglePass(intermediate, adjacency_matrix_, relu);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(0), result(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0), result(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(-0.08), result(2), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.14), result(3), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.04), result(4), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.38), result(5), tolerance);
}

TEST_F(GraphFixture, Predict_Regression)
{
    Vector<2> input;
    input << 1, 2;

    auto result = graph_.Predict(input);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(0.16), result(0), tolerance);
}
}  // namespace
