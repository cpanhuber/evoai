#ifndef EVOAI__EVOLUTION__POPULATION_H
#define EVOAI__EVOLUTION__POPULATION_H

#include <evoai/common/types.h>
#include <evoai/graph/graph.h>

#include <random>
#include <vector>

namespace evoai
{

namespace detail
{
template <typename GraphType>
struct Specimen
{
    using Traits = typename GraphType::Traits;

    ValueType mutancy;
    typename Traits::AdjacencyType adjacency;
};

template <IndexType TotalN, typename ActivationFunctionType>
struct ActivationTracker : public ActivationFunctionType
{
    template <typename Derived>
    auto operator()(MatrixBase<Derived> const& x)
    {
        auto activation = ActivationFunctionType::operator()(x);
        counted_activations = counter(counted_activations, activation);
        return activation;
    }
    Vector<TotalN> counted_activations;
    aggregation::Counter aggregator;
};

template <typename Derived>
auto CreateWeightMutationFactors(MatrixBase<Derived> const& accumulated_activations, ValueType const mutancy)
{
    return (((accumulated_activations / accumulated_activations.mean()).array() - static_cast<ValueType>(1.0)) *
            mutancy) +
           static_cast<ValueType>(1.0);
}

template <typename DerivedAdjacency, typename DerivedMutationFactors>
void MutateAdjacency(MatrixBase<DerivedAdjacency>& adjacency, MatrixBase<DerivedMutationFactors> const& mutation)
{
    auto rows = Vector<adjacency.RowsAtCompileTime>::Ones();
    auto cwise_mutation_factors = (rows * mutation.transpose()).array();
    adjacency = (adjacency.array() * cwise_mutation_factors).matrix();
}

template <typename RandomGenerator>
ValueType SampleNormal(ValueType const mean, ValueType const stddev, RandomGenerator& generator)
{
    std::normal_distribution<ValueType> distribution{mean, stddev};
    return distribution(generator);
}

template <typename GraphType, typename Derived, typename RandomGenerator>
void Mutate(Specimen<GraphType>& specimen,
            MatrixBase<Derived> const& accumulated_activations,
            RandomGenerator& generator,
            ValueType const mutancy_change_deviation)
{
    auto weight_mutation_factors = CreateWeightMutationFactors(accumulated_activations, specimen.mutancy);
    MutateAdjacency(specimen.adjacency, weight_mutation_factors);
    specimen.mutancy *=
        SampleNormal(static_cast<ValueType>(1.0), static_cast<ValueType>(mutancy_change_deviation), generator);
}

}  // namespace detail
template <typename GraphType>
using Population = std::vector<detail::Specimen<GraphType>>;

}  // namespace evoai

#endif  // EVOAI__EVOLUTION__POPULATION_H
