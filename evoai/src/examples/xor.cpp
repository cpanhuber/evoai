#include <evoai/common/types.h>
#include <evoai/evolution/evolution.h>
#include <evoai/graph/activation/log_softmax.h>
#include <evoai/graph/activation/relu.h>
#include <evoai/graph/graph.h>
#include <evoai/graph/loss/cross_entropy.h>

#include <iostream>

int main(int argc, char** argv)
{
    std::vector<evoai::Vector<2>> inputs;
    std::vector<evoai::Vector<2>> truths;
    inputs.push_back({0.0, 0.0});
    inputs.push_back({0.0, 1.0});
    inputs.push_back({1.0, 0.0});
    inputs.push_back({1.0, 1.0});
    truths.push_back({1.0, 0.0});
    truths.push_back({0.0, 1.0});
    truths.push_back({0.0, 1.0});
    truths.push_back({1.0, 0.0});

    using GraphType = evoai::ClassificationGraph<2, 2, 5, 2>;

    evoai::Evolution<GraphType, evoai::loss::CrossEntropy, evoai::mutation::Tingri> evolution;

    GraphType graph = evolution.Evolve(inputs, truths, true);

    std::cout << "Predict " << graph.Predict({0, 0})(1) << " from 0, 0" << std::endl;
    std::cout << "Predict " << graph.Predict({1, 0})(1) << " from 1, 0" << std::endl;
    std::cout << "Predict " << graph.Predict({0, 1})(1) << " from 0, 1" << std::endl;
    std::cout << "Predict " << graph.Predict({1, 1})(1) << " from 1, 1" << std::endl;

    return 0;
}
