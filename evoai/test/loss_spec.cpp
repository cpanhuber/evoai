#include <evoai/graph/loss/cross_entropy.h>
#include <evoai/graph/loss/log_cross_entropy.h>
#include <evoai/graph/loss/mean_squared_error.h>

#include <evoai/common/types.h>

#include <gtest/gtest.h>

namespace
{

using namespace evoai;

TEST(MeanSquaredError, WhenTypical)
{
    loss::MeanSquaredError mse;
    Vector<3> y_predicted;
    y_predicted << -1.0, 2.0, 1.0;
    Vector<3> y_actual;
    y_actual << 1.0, 2.0, 1.5;

    auto result = mse(y_predicted, y_actual);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(4.25 / 3.0), result, tolerance);
}

TEST(CrossEntropy, WhenTypical)
{
    loss::CrossEntropy cross_entropy;
    Vector<5> y_predicted;
    y_predicted << 0.1, 0.5, 0.1, 0.1, 0.2;
    Vector<5> y_actual;
    y_actual << 1.0, 0.0, 0.0, 0.0, 0.0;

    auto result = cross_entropy(y_predicted, y_actual);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(2.302585), result, tolerance);
}

TEST(LogCrossEntropy, WhenTypical)
{
    loss::LogCrossEntropy log_cross_entropy;
    Vector<5> y_predicted;
    y_predicted << 0.1, 0.5, 0.1, 0.1, 0.2;
    Vector<5> y_actual;
    y_actual << 1.0, 0.0, 0.0, 0.0, 0.0;

    auto result = log_cross_entropy(y_predicted.array().log().matrix(), y_actual);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(2.302585), result, tolerance);
}

}  // namespace
