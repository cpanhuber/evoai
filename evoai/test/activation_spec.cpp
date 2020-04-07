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

}  // namespace
