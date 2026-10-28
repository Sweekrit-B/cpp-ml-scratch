# pragma once
# include "Matrix.hpp"
# include <utility>
# include <tuple>
# include <functional>

// a struct bundles several related values into one type
// members are PUBLIC by default instead of being private
struct Activation {
    std::function<Matrix(const Matrix&)>forward; // the function's forward pass
    std::function<Matrix(const Matrix&)>backward; // the function's backward pass
};

// a namespace groups names under a label so they do not collide with other code
namespace Activations {
    Activation relu();
    Activation tanh();
}

class MultilayerPerceptron {
    private:
        std::vector<Matrix> weights; // weight matrices for each layer
        std::vector<Activation> activations; // activation functions for each layer
        // we need to cache forward pass results for backpropagation
        std::vector<Matrix> layerInputs; // inputs to each layer (input to each layer)
        std::vector<Matrix> preActivations; // outputs of each layer (Z for each layer)

    public:
        static size_t countClasses(const Matrix& raw, size_t numCols); // counts distinct classes in the last column of the raw matrix
        static std::tuple<Matrix, Matrix, Matrix> prepare(const Matrix& raw, size_t numCols, size_t numClasses); // prepares the data for training
        static Matrix oneHotEncode(const Matrix& y, size_t numClasses); // converts labels to one-hot encoding
        MultilayerPerceptron(const std::vector<size_t>& layerSizes,
            const std::vector<Activation>& activations);
        Matrix forwardMLP(const Matrix& input); // forward pass through the network
        void backwardMLP(const Matrix& target, double learningRate, double lambda); // backward pass and weight update (must follow forwardMLP on the same batch)
        void trainMLP(const Matrix& trainX, const Matrix& trainY, double lambda, double learningRate, int maxIterations, int batchSize); // training loop
        Matrix predictMLP(const Matrix& input); // make predictions on new data
        double evaluateMLPCE(const Matrix& testX, const Matrix& testY); // average cross-entropy loss; testY is one-hot (n x K)
        double evaluateMLPAccuracy(const Matrix& testX, const Matrix& testY); // fraction classified correctly; testY is class labels (n x 1)
};