#include <evoai/graph/aggregation/accumulator.h>
#include <evoai/graph/aggregation/counter.h>

#include <evoai/common/types.h>

#include <gtest/gtest.h>

namespace
{

using namespace evoai;

TEST(Accumulator, CreateInitial_WhenTypical)
{
    aggregation::Accumulator accumulator{};
    auto initial = aggregation::CreateInitial<3>(accumulator);

    EXPECT_EQ(0, initial(0));
    EXPECT_EQ(0, initial(1));
    EXPECT_EQ(0, initial(2));
}

TEST(Accumulator, call_WhenTypical)
{
    aggregation::Accumulator accumulator{};
    Vector<3> a;
    a << -1.0, 0.0, 1.0;
    Vector<3> b;
    b << -2.0, 1.0, -1.0;

    auto result = accumulator(a, b);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(-3.0), result(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(0.0), result(2), tolerance);
}

TEST(Counter, CreateInitial_WhenTypical)
{
    aggregation::Counter counter{};
    auto initial = aggregation::CreateInitial<3>(counter);

    EXPECT_EQ(0, initial(0));
    EXPECT_EQ(0, initial(1));
    EXPECT_EQ(0, initial(2));
}

TEST(Counter, call_WhenTypical)
{
    aggregation::Counter counter{};
    Vector<3> a;
    a << 2.0, 0.0, 1.0;
    Vector<3> b;
    b << 0.1, 0.2, 0.0;

    auto result = counter(a, b);

    auto tolerance = static_cast<ValueType>(1e-5);
    EXPECT_NEAR(static_cast<ValueType>(3.0), result(0), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(1), tolerance);
    EXPECT_NEAR(static_cast<ValueType>(1.0), result(2), tolerance);
}

}  // namespace
