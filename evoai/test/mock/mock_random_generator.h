#ifndef EVOAI__TEST__MOCK__MOCK_RANDOM_GENERATOR_H
#define EVOAI__TEST__MOCK__MOCK_RANDOM_GENERATOR_H

#include <evoai/common/types.h>

#include <climits>

namespace evoai
{
namespace test
{

// Since implementation of distributions such as normal_distribution
// are not defined, using this may still give different results on
// different platforms. It is recommended to obtain the output
// of a distribution used in a test with this generator and calculate
// the test expectation from this output
struct MockRandomGenerator
{
    using result_type = uint32_t;

    constexpr result_type min()
    {
        return std::numeric_limits<result_type>::lowest();
    }

    constexpr result_type max()
    {
        return std::numeric_limits<result_type>::max();
    }

    result_type operator()()
    {
        return output;
    }

    result_type output = 1234;
};
}  // namespace test
}  // namespace evoai

#endif  // EVOAI__TEST__MOCK__MOCK_RANDOM_GENERATOR_H
