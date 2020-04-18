#ifndef EVOAI__EVOLUTION__MUTATION__TINGRI_H
#define EVOAI__EVOLUTION__MUTATION__TINGRI_H

#include <evoai/common/types.h>
#include <evoai/evolution/population.h>

#include <unsupported/Eigen/SpecialFunctions>

#include <cmath>
#include <random>

// Tingri: Town in Tibet, often used as base for climbers attempting
// to ascend Mount Everest.
// https://en.wikipedia.org/wiki/Tingri_(town)
// The Tingri strategy involves changing weights
// and spawning and killing connections with a chance proportional
// to the passed accumulated activation vector

namespace evoai
{
namespace mutation
{
namespace detail
{
template <typename Derived>
auto CreateWeightMutationFactors(MatrixBase<Derived> const& accumulated_activations, ValueType const mutancy)
{
    return ((((accumulated_activations / accumulated_activations.mean()).array() - static_cast<ValueType>(1.0)) *
             mutancy) +
            static_cast<ValueType>(1.0))
        .matrix();
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

template <typename Derived>
auto NormalCDF(MatrixBase<Derived> const& x)
{
    return ((((x * -1) * M_SQRT1_2).array().erfc()) * 0.5).matrix();
}

template <typename CDFDerived, typename OnesDerived>
auto CreateKillChances(MatrixBase<CDFDerived> const& normal_cdf,
                       MatrixBase<OnesDerived> const& n_ones,
                       ValueType const kill_prior)
{
    return n_ones * ((n_ones - normal_cdf) * kill_prior).transpose();
}

template <typename CDFDerived, typename OnesDerived>
auto CreateReviveChances(MatrixBase<CDFDerived> const& normal_cdf,
                         MatrixBase<OnesDerived> const& n_ones,
                         ValueType const revive_prior)
{
    return n_ones * ((normal_cdf)*revive_prior).transpose();
}

template <typename ChancesDerived, typename RandomGenerator>
auto CreateKillMatrix(MatrixBase<ChancesDerived> const& kill_chances, RandomGenerator& generator)
{
    return kill_chances.unaryExpr([&](ValueType x) {
        std::uniform_real_distribution<ValueType> uniform{static_cast<ValueType>(0.0), static_cast<ValueType>(1.0)};
        return uniform(generator) < x ? static_cast<ValueType>(0) : static_cast<ValueType>(1);
    });
}

template <typename DerivedRevive, typename DerivedAdjacency, typename RandomGenerator>
auto CreateReviveMatrix(MatrixBase<DerivedRevive> const& revive_chances,
                        MatrixBase<DerivedAdjacency> const& adjacency,
                        RandomGenerator& generator)
{
    return revive_chances.binaryExpr(adjacency, [&](ValueType revive_scalar, ValueType adjacency_scalar) {
        std::uniform_real_distribution<ValueType> uniform{static_cast<ValueType>(0.0), static_cast<ValueType>(1.0)};
        std::normal_distribution<ValueType> normal{static_cast<ValueType>(0.0), static_cast<ValueType>(0.5)};
        return std::abs(adjacency_scalar) < std::numeric_limits<ValueType>::min() && uniform(generator) < revive_scalar
                   ? static_cast<ValueType>(normal(generator))
                   : static_cast<ValueType>(0);
    });
}

template <typename DerivedAdjacency, typename DerivedDeviations, typename RandomGenerator>
void ReviveOrKill(MatrixBase<DerivedAdjacency>& adjacency,
                  MatrixBase<DerivedDeviations> const& deviations,
                  ValueType const revive_prior,
                  ValueType const kill_prior,
                  RandomGenerator& generator)
{
    auto normal_cdf = NormalCDF(deviations);
    auto n_ones = Vector<adjacency.RowsAtCompileTime>::Ones();
    auto kill_chances = CreateKillChances(normal_cdf, n_ones, kill_prior);
    auto revive_chances = CreateReviveChances(normal_cdf, n_ones, revive_prior);
    auto kill_matrix = CreateKillMatrix(kill_chances, generator);
    auto revive_matrix = CreateReviveMatrix(revive_chances, adjacency, generator);

    adjacency = (adjacency.array() * kill_matrix.array()).matrix() + revive_matrix;
}

template <typename DerivedActivations>
auto CreateStandardDeviations(MatrixBase<DerivedActivations> const& activations)
{
    auto deviations = (activations.array() - activations.mean());
    return (deviations / deviations.abs().mean()).matrix();
}

}  // namespace detail
struct Tingri
{
    struct Settings
    {
        ValueType mutancy_change_deviation = static_cast<ValueType>(0.01);
        ValueType revive_change_deviation = static_cast<ValueType>(0.01);
        ValueType kill_change_deviation = static_cast<ValueType>(0.01);

        ValueType mutancy_initial_mean = static_cast<ValueType>(0.0);
        ValueType revive_initial_mean = static_cast<ValueType>(0.0);
        ValueType kill_initial_mean = static_cast<ValueType>(0.0);

        ValueType weight_initial_mean = static_cast<ValueType>(0.0);
        ValueType weight_initial_deviation = static_cast<ValueType>(1.0);
    };

    struct Properties
    {
        ValueType mutancy;
        ValueType revive_prior;
        ValueType kill_prior;
    };

    template <typename GraphType, typename Derived, typename RandomGenerator>
    void operator()(evoai::detail::Specimen<GraphType, Properties>& specimen,
                    MatrixBase<Derived> const& accumulated_activations,
                    RandomGenerator& generator) const
    {
        auto weight_mutation_factors = detail::CreateWeightMutationFactors(accumulated_activations, specimen.mutancy);
        detail::MutateAdjacency(specimen.adjacency, weight_mutation_factors);
        auto deviations = detail::CreateStandardDeviations(accumulated_activations);
        detail::ReviveOrKill(specimen.adjacency, deviations, specimen.revive_prior, specimen.kill_prior, generator);
        specimen.mutancy = detail::SampleNormal(specimen.mutancy, settings.mutancy_change_deviation, generator);
        specimen.revive_prior =
            detail::SampleNormal(specimen.revive_prior, settings.revive_change_deviation, generator);
        specimen.kill_prior = detail::SampleNormal(specimen.kill_prior, settings.kill_change_deviation, generator);
    }

    template <typename GraphType, typename RandomGenerator>
    evoai::detail::Specimen<GraphType, Properties> CreateSpecimen(RandomGenerator& generator) const
    {
        evoai::detail::Specimen<GraphType, Properties> specimen;
        specimen.mutancy =
            detail::SampleNormal(settings.mutancy_initial_mean, settings.mutancy_change_deviation, generator);
        specimen.kill_prior =
            detail::SampleNormal(settings.kill_initial_mean, settings.kill_change_deviation, generator);
        specimen.revive_prior =
            detail::SampleNormal(settings.revive_initial_mean, settings.revive_change_deviation, generator);

        using AdjacencyType = typename GraphType::Traits::AdjacencyType;
        specimen.adjacency =
            AdjacencyType::NullaryExpr(AdjacencyType::RowsAtCompileTime, AdjacencyType::ColsAtCompileTime, [&]() {
                return detail::SampleNormal(settings.weight_initial_mean, settings.weight_initial_deviation, generator);
            });
        return specimen;
    }

    Settings settings;
};
}  // namespace mutation
}  // namespace evoai
#endif  //  EVOAI__EVOLUTION__MUTATION__TINGRI_H
