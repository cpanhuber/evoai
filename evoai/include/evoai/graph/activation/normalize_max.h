#ifndef EVOAI__GRAPH__ACTIVATION__NORMALIZE_MAX_H
#define EVOAI__GRAPH__ACTIVATION__NORMALIZE_MAX_H

#include <evoai/common/types.h>
#include <evoai/graph/activation/activator.h>

#include <cmath>
#include <utility>

namespace evoai
{

namespace activation
{

struct NormalizeMax : public detail::Activator
{
    template <typename Derived>
    auto operator()(MatrixBase<Derived> const& in)
    {
        return in / in.maxCoeff();
    }
};
}  // namespace activation

}  // namespace evoai

#endif  // EVOAI__GRAPH__ACTIVATION__NORMALIZE_MAX_H

