#ifndef EVOAI__GRAPH__GRAPH_H
#define EVOAI__GRAPH__GRAPH_H

#include <evoai/common/types.h>
#include <evoai/graph/aggregation/aggregator.h>

#include <cstddef>

namespace evoai
{

namespace detail
{
template <typename DerivedIn, typename DerivedAdjacency, typename ActivationFunctionType>
auto SinglePass(MatrixBase<DerivedIn> const& input,
                MatrixBase<DerivedAdjacency> const& adjacency_matrix,
                ActivationFunctionType& activation_function)
{
    return activation_function(adjacency_matrix * input);
}

template <IndexType Iteration,
          IndexType InputN,
          IndexType OutputN,
          IndexType TotalN,
          typename DerivedInput,
          typename DerivedIntermediate,
          typename DerivedAdjacency,
          typename DerivedAggregated,
          typename ActivationFunctionType,
          typename AggregatorType,
          typename TransformatorType>
struct PredictCycle
{
    auto operator()(MatrixBase<DerivedInput> const& input,
                    MatrixBase<DerivedIntermediate>& intermediate,
                    MatrixBase<DerivedAdjacency> const& adjacency,
                    MatrixBase<DerivedAggregated> const& aggregated_output,
                    ActivationFunctionType& activation,
                    AggregatorType& aggregator,
                    TransformatorType& transformator)
    {
        auto intermediate_with_input = intermediate + input;
        auto next_intermediate = detail::SinglePass(intermediate_with_input, adjacency, activation);
        auto next_aggregated =
            aggregator(aggregated_output, next_intermediate.template block<OutputN, 1>(TotalN - OutputN, 0));

        using DerivedIntermediateNext = get_derived_t<std::decay_t<decltype(next_intermediate)>>;
        using DerivedAggregatedNext = get_derived_t<std::decay_t<decltype(next_aggregated)>>;

        return PredictCycle<Iteration - 1,
                            InputN,
                            OutputN,
                            TotalN,
                            DerivedInput,
                            DerivedIntermediateNext,
                            DerivedAdjacency,
                            DerivedAggregatedNext,
                            ActivationFunctionType,
                            AggregatorType,
                            TransformatorType>{}(
            input, next_intermediate, adjacency, next_aggregated, activation, aggregator, transformator);
    }
};

template <IndexType InputN,
          IndexType OutputN,
          IndexType TotalN,
          typename DerivedInput,
          typename DerivedIntermediate,
          typename DerivedAdjacency,
          typename DerivedAggregated,
          typename ActivationFunctionType,
          typename AggregatorType,
          typename TransformatorType>
struct PredictCycle<0,
                    InputN,
                    OutputN,
                    TotalN,
                    DerivedInput,
                    DerivedIntermediate,
                    DerivedAdjacency,
                    DerivedAggregated,
                    ActivationFunctionType,
                    AggregatorType,
                    TransformatorType>
{
    auto operator()(MatrixBase<DerivedInput> const& input,
                    MatrixBase<DerivedIntermediate>& intermediate,
                    MatrixBase<DerivedAdjacency> const& adjacency,
                    MatrixBase<DerivedAggregated> const& aggregated_output,
                    ActivationFunctionType& activation,
                    AggregatorType& aggregator,
                    TransformatorType& transformator)
    {
        return transformator(aggregated_output);
    }
};

template <typename GraphTraits>
typename GraphTraits::OutputType Predict(typename GraphTraits::InputType const& input,
                                         typename GraphTraits::AdjacencyType const& adjacency)
{
    auto activator = typename GraphTraits::ActivationFunctionType{};
    auto aggregator = typename GraphTraits::AggregatorType{};
    auto transformator = typename GraphTraits::TransformatorType{};

    Vector<GraphTraits::k_total_neurons> scaled_input = Vector<GraphTraits::k_total_neurons>::Zero();
    scaled_input.template block<GraphTraits::k_input_neurons, 1>(0, 0) = input;
    auto intermediate = Vector<GraphTraits::k_total_neurons>::Zero();
    auto aggregated = aggregation::CreateInitial<GraphTraits::k_output_neurons>(aggregator);

    using DerivedInput = get_derived_t<std::decay_t<decltype(scaled_input)>>;
    using DerivedIntermediate = get_derived_t<std::decay_t<decltype(intermediate)>>;
    using DerivedAdjacency = get_derived_t<std::decay_t<decltype(adjacency)>>;
    using DerivedAggregated = get_derived_t<std::decay_t<decltype(aggregated)>>;

    return PredictCycle<GraphTraits::k_forward_iterations,
                        GraphTraits::k_input_neurons,
                        GraphTraits::k_output_neurons,
                        GraphTraits::k_total_neurons,
                        DerivedInput,
                        DerivedIntermediate,
                        DerivedAdjacency,
                        DerivedAggregated,
                        typename GraphTraits::ActivationFunctionType,
                        typename GraphTraits::AggregatorType,
                        typename GraphTraits::TransformatorType>{}(
        scaled_input, intermediate, adjacency, aggregated, activator, aggregator, transformator);
}
}  // namespace detail

template <IndexType InputN,
          IndexType OutputN,
          IndexType HiddenN,
          IndexType ForwardIterations,
          typename ActivationFunctionT,
          typename AggregatorT,
          typename TransformatorT>
class GraphTraits
{
  public:
    static constexpr IndexType k_input_neurons = InputN;
    static constexpr IndexType k_output_neurons = OutputN;
    static constexpr IndexType k_hidden_neurons = HiddenN;
    static constexpr IndexType k_total_neurons = k_input_neurons + k_output_neurons + k_hidden_neurons;
    static constexpr IndexType k_forward_iterations = ForwardIterations;

    using ActivationFunctionType = ActivationFunctionT;
    using AggregatorType = AggregatorT;
    using TransformatorType = TransformatorT;
    using InputType = Vector<InputN>;
    using OutputType = Vector<OutputN>;
    using AdjacencyType = Matrix<k_total_neurons, k_total_neurons>;

    // static_assert(k_input_neurons > 0, "Graph has to have a source, InputN has to be > 0");
    // static_assert(k_output_neurons > 0, "Graph has to have a sink, OutputN has to be > 0");
    // static_assert(k_hidden_neurons >= 0, "Number of hidden neurons has to be positive");
    // static_assert(ActivationFunctionType::k_is_activation_function,
    //               "ActivationFunctionType is not a valid Activation Function");
    // static_assert(AggregatorType::k_is_aggregator, "AggregatorType is not a valid Aggregator");
    // static_assert(TransformatorType::k_is_transformator, "TransformatorType is not a valid Transformator");
};

template <IndexType InputN,
          IndexType OutputN,
          IndexType HiddenN,
          IndexType ForwardIterations,
          typename ActivationFunctionType,
          typename AggregatorType,
          typename TransformatorType>
class NeuralGraph : public GraphTraits<InputN,
                                       OutputN,
                                       HiddenN,
                                       ForwardIterations,
                                       ActivationFunctionType,
                                       AggregatorType,
                                       TransformatorType>
{
  public:
    using Traits = GraphTraits<InputN,
                               OutputN,
                               HiddenN,
                               ForwardIterations,
                               ActivationFunctionType,
                               AggregatorType,
                               TransformatorType>;
    NeuralGraph(){};
    NeuralGraph(typename Traits::AdjacencyType const& adjacency_matrix) : adjacency_matrix_{adjacency_matrix} {}

    NeuralGraph(NeuralGraph const&) = default;
    NeuralGraph(NeuralGraph&&) = default;
    ~NeuralGraph() = default;

    NeuralGraph& operator=(NeuralGraph const&) = default;
    NeuralGraph& operator=(NeuralGraph&&) = default;

    auto Predict(Vector<Traits::k_input_neurons> const& input) const
    {
        return detail::Predict<Traits>(input, adjacency_matrix_);
    }

  private:
    typename Traits::AdjacencyType adjacency_matrix_;
};

}  // namespace evoai

#endif  // ifndef EVOAI__GRAPH__GRAPH_H
