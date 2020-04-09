#include <evoai/graph/transformation/softmax.h>

#include <evoai/common/types.h>

#include <gtest/gtest.h>

#include <cmath>

namespace
{

using namespace evoai;

TEST(LogSoftmax, call_WhenTypical)
{
    transformation::LogSoftmax log_softmax;
    Vector<3> x;
    x << 3.0, 1.0, 0.2;

    auto result = log_softmax(x);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(std::log(0.8360188)), result(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(std::log(0.11314284)), result(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(std::log(0.05083836)), result(2), tolerance);
}

}  // namespace
