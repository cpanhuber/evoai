#ifndef EVOAI__GRAPH__GRAPH_H
#define EVOAI__GRAPH__GRAPH_H

#include <evoai/common/types.h>

#include <cstddef>

namespace evoai
{

namespace detail
{
template <IndexType N, typename ActivationFunctionType>
Vector<N> SinglePass(Vector<N> const& input,
                     Matrix<N, N> const& adjacency_matrix,
                     ActivationFunctionType& activation_function)
{
    return activation_function(adjacency_matrix * input);
}

template <IndexType N, IndexType M>
void AddVectorBlockAt(Vector<N>& result, Vector<M> const& input, IndexType const index)
{
    static_assert(N > M, "Trying to add a larger vector to a smaller");
    result.block<M, 1>(index, 0) += input;
}

}  // namespace detail

template <IndexType InputN, IndexType OutputN, IndexType HiddenN, typename ActivationFunctionType>
class NeuralGraph
{
  public:
    static constexpr IndexType k_input_neurons = InputN;
    static constexpr IndexType k_output_neurons = OutputN;
    static constexpr IndexType k_hidden_neurons = HiddenN;
    static constexpr IndexType k_total_neurons = k_input_neurons + k_output_neurons + k_hidden_neurons;

    NeuralGraph() {}

    NeuralGraph(NeuralGraph const&) = default;
    NeuralGraph(NeuralGraph&&) = default;
    ~NeuralGraph() = default;

    NeuralGraph& operator=(NeuralGraph const&) = delete;
    NeuralGraph& operator=(NeuralGraph&&) = delete;

    Vector<k_output_neurons> Predict(Vector<k_input_neurons> const& input, IndexType iterations) const
    {
        auto intermediate = Vector<k_total_neurons>::Zeros();
        for (IndexType i = 0; i < iterations; ++i)
        {
            detail::AddVectorBlockAt(intermediate, input, 0);

            intermediate = detail::SinglePass(intermediate, adjacency_matrix_, activation_function_);
        }
        return intermediate.block<k_output_neurons, 1>(k_total_neurons - k_output_neurons, 0);
    }

  private:
    Matrix<k_total_neurons, k_total_neurons> adjacency_matrix_;
    ActivationFunctionType activation_function_;
};
}  // namespace evoai

#endif  // ifndef EVOAI__GRAPH__GRAPH_H
