# include "MultilayerPerceptron.hpp"

# include <iostream>
# include <set>
# include <stdexcept>
# include <random>

Activation Activations::relu() {
    return Activation{
        [](const Matrix& x) { return x.relu(); },
        [](const Matrix& x) { return x.reluDerivative(); }
    };
};

Activation Activations::tanh() {
    return Activation{
        [](const Matrix& x) { return x.tanh(); },
        [](const Matrix& x) { return x.tanhDerivative(); }
    };
};

size_t MultilayerPerceptron::countClasses(const Matrix& raw, size_t numCols) {
    std::set<size_t> classes;
    for (size_t i = 0; i < raw.numRows(); ++i) {
        classes.insert(static_cast<size_t>(raw(i, numCols)));
    }
    return classes.size();
};

Matrix MultilayerPerceptron::oneHotEncode(const Matrix& y, size_t numClasses) {
    size_t n = y.numRows();
    Matrix Y(n, numClasses);
    for (size_t i = 0; i < n; ++i) {
        size_t classIndex = static_cast<size_t>(y(i, 0));
        if (classIndex >= numClasses) {
            throw std::out_of_range("Class index out of range in one-hot encoding.");
        }
        Y(i, classIndex) = 1.0;
    }
    return Y;
};

std::tuple<Matrix, Matrix, Matrix> MultilayerPerceptron::prepare(const Matrix& raw, size_t numCols, size_t numClasses) {
    size_t n = raw.numRows();
    Matrix X(n, numCols);
    Matrix y(n, 1);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            X(i, j) = raw(i, j);
        }
        y(i, 0) = raw(i, numCols);
    }
    Matrix Y = oneHotEncode(y, numClasses); // convert to one-hot encoding
    return {X, y, Y};
};

MultilayerPerceptron::MultilayerPerceptron(const std::vector<size_t>& layerSizes,
                                         const std::vector<Activation>& activations)
    : activations(activations) {
    if (layerSizes.size() < 2) { // must have at least input and output
        throw std::invalid_argument("At least two layers (input and output) are required.");
    }
    if (activations.size() != layerSizes.size() - 2) { // must have one activation function per layer except the input layer
        throw std::invalid_argument("Number of activation functions must be one per hidden layer.");
    }

    std::mt19937 rng(std::random_device{}()); // random number generator

    for (size_t l = 0; l + 1 < layerSizes.size(); ++l) { // for every layer except the last one
        size_t inputSize = layerSizes[l] + 1; // +1 for bias
        size_t outputSize = layerSizes[l + 1]; // next layer's size
        Matrix weightMatrix(inputSize, outputSize);

        // Initialize weights with small random values
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / layerSizes[l]));
        for (size_t i = 0; i < inputSize; ++i) {
            for (size_t j = 0; j < outputSize; ++j) {
                weightMatrix(i, j) = dist(rng);
            }
        }
        weights.push_back(weightMatrix);
    }
};

Matrix MultilayerPerceptron::forwardMLP(const Matrix& input) {
    // example:
        // step 1: append bias column to input
        // step 2: compute Z = X * W for the first layer
        // step 3: apply activation function to Z to get A
        // step 4: repeat for each layer, using the output A of the previous layer as input to the next layer
        // step 5: do a softmax on the final output layer to get probabilities

    Matrix currentInput = input.appendBiasColumn(); // step 1: append bias column to input
    layerInputs.clear(); // clear previous cached inputs
    preActivations.clear(); // clear previous cached pre-activations
    for (size_t l = 0; l < weights.size(); ++l) { // for each layer
        layerInputs.push_back(currentInput); // cache input to the first layer
        Matrix preActivation = currentInput * weights[l]; // step 2: compute Z = X * W for the first layer
        preActivations.push_back(preActivation); // cache pre-activation
        if (l + 1 < weights.size()) { // if not the last layer
            currentInput = activations[l].forward(preActivation).appendBiasColumn(); // append bias column for next layer
        } else {
            return preActivation.softmax(); // step 5: do a softmax on the final output layer to get probabilities
        }
    }
    throw std::logic_error("forwardMLP should have returned before reaching this point.");
}

void MultilayerPerceptron::backwardMLP(const Matrix& target, double learningRate, double lambda) {
    // step 1: initial delta_2: P - Y
    Matrix P = preActivations.back().softmax();
    Matrix delta = P - target;

    std::vector<Matrix> grads;
    grads.resize(weights.size(), Matrix(1, 1));

    for (size_t l = weights.size(); l-- > 0; ) { // for each of the remaining layers
        grads[l] = layerInputs[l].transpose() * delta;
        if (l > 0) {
            Matrix back = (delta * weights[l].transpose()).removeLastColumn();
            delta = back.hadamardProduct(activations[l - 1].backward(preActivations[l - 1])); // applies the derivative of the activation function to the backpropagated error
        }
    }

    for (size_t l = 0; l < weights.size(); ++l) {
        // average the data gradient over the batch, add the L2 term, then step
        Matrix gradient = grads[l] * (1.0 / target.numRows()) + weights[l] * lambda;
        weights[l] = weights[l] - gradient * learningRate; // update weights
    }
}

void MultilayerPerceptron::trainMLP(const Matrix& trainX, const Matrix& trainY, double lambda, double learningRate, int maxIterations, int batchSize) {
    const size_t numTrainRows = trainX.numRows();

    for (int step = 0; step < maxIterations; ++step) {
        std::vector<size_t> batchIndices = Matrix::sampleRowIndices(numTrainRows, static_cast<size_t>(batchSize));
        Matrix xBatch = trainX.selectRows(batchIndices);
        Matrix yBatch = trainY.selectRows(batchIndices);
        std::cout << "Step " << step + 1 << ": Sampled a batch of rows for stochastic gradient descent." << std::endl;

        forwardMLP(xBatch);
        backwardMLP(yBatch, learningRate, lambda);

        double loss = evaluateMLPCE(xBatch, yBatch);
        if (step % 10 == 0) {
            std::cout << "Step " << step + 1 << ": Loss = " << loss << std::endl;
        }
    }

    return;
}

Matrix MultilayerPerceptron::predictMLP(const Matrix& input) {
    return forwardMLP(input);
}

double MultilayerPerceptron::evaluateMLPCE(const Matrix& testX, const Matrix& testY) {
    Matrix predictions = predictMLP(testX);
    double loss = 0.0;
    for (size_t i = 0; i < testY.numRows(); ++i) {
        for (size_t j = 0; j < testY.numCols(); ++j) {
            double yTrue = testY(i, j);
            double yPred = predictions(i, j);
            double ce = -yTrue * std::log(yPred + 1e-15); // add small epsilon to avoid log(0)
            loss += ce;
        }
    }
    return loss / testY.numRows(); // average loss over all samples
}

double MultilayerPerceptron::evaluateMLPAccuracy(const Matrix& testX, const Matrix& testY) {
    Matrix predictions = predictMLP(testX);
    size_t correctCount = 0;
    for (size_t i = 0; i < testY.numRows(); ++i) {
        size_t predictedClass = 0;
        double maxProb = predictions(i, 0);
        for (size_t j = 1; j < predictions.numCols(); ++j) {
            if (predictions(i, j) > maxProb) {
                maxProb = predictions(i, j);
                predictedClass = j;
            }
        }
        if (predictedClass == static_cast<size_t>(testY(i, 0))) {
            correctCount++;
        }
    }
    return static_cast<double>(correctCount) / testY.numRows(); // accuracy
}