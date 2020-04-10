#include <evoai/graph/activation/log_softmax.h>
#include <evoai/graph/activation/normalize_max.h>
#include <evoai/graph/activation/relu.h>

#include <evoai/common/types.h>

#include <gtest/gtest.h>

namespace
{

using namespace evoai;

TEST(RelU, call_WhenInactiveAmbivalentActive)
{
    activation::RelU relu;
    Vector<3> x;
    x << -1.0, 0.0, 1.0;

    auto result = relu(x);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(0.0), result(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.0), result(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(2), tolerance);
}

TEST(LogSoftmax, call_WhenTypical)
{
    activation::LogSoftmax log_softmax;
    Vector<3> x;
    x << 3.0, 1.0, 0.2;

    auto result = log_softmax(x);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(std::log(0.8360188)), result(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(std::log(0.11314284)), result(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(std::log(0.05083836)), result(2), tolerance);
}

TEST(NormalizeMax, call_WhenTypical)
{
    activation::NormalizeMax normalize_max;
    Vector<3> x;
    x << 3.0, 1.0, 0.2;

    auto result = normalize_max(x);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(3.0 / 3.0), result(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0 / 3.0), result(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.2 / 3.0), result(2), tolerance);
}

}  // namespace
