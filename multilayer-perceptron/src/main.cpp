# include "Matrix.hpp"
# include <stdexcept>
# include "DataLoader.hpp"
# include "MultilayerPerceptron.hpp"
# include <iostream>
# include <chrono>

// Orchestrates the pipeline for one dataset: load -> split once -> prepare -> train -> evaluate.
double runMultilayerPerceptron(const std::string& filename, int batchSize, double lambda = 0,
                               double learningRate = 0.1, int maxIterations = 1000,
                               double tolerance = 1e-6, double testRatio = 0.2,
                               const std::vector<size_t>& hiddenSizes = {16, 32, 16}, // hidden layer widths; input/output sizes come from the data
                               const std::vector<Activation>& activations = {Activations::relu(), Activations::relu(), Activations::relu()}) { // one per hidden layer
    // step 0 - receive some data as input
    auto loadStart = std::chrono::steady_clock::now();
    Matrix data = DataLoader::loadCSV(filename);
    auto loadEnd = std::chrono::steady_clock::now();
    std::cout << "[timing] loadCSV: " << std::chrono::duration<double>(loadEnd - loadStart).count() << "s" << std::endl;
    const size_t numCols = data.numCols() - 1; // last column is the target value
    const size_t numClasses = MultilayerPerceptron::countClasses(data, numCols);
    std::cout << "Loaded data with " << data.numRows() << " rows and " << numCols + 1 << " columns." << std::endl;

    // step 0.5 - split into train/test sets BEFORE separating features from the target.
    // Matrix::trainTestSplit shuffles internally each call, so splitting the raw
    // (features + target still combined) matrix once — here, and only here — keeps
    // every row's features paired with its own label in both resulting sets.
    std::vector<Matrix> split = data.trainTestSplit(testRatio);
    Matrix trainRaw = split[0];
    Matrix testRaw = split[1];
    std::cout << "Split into " << trainRaw.numRows() << " training rows and " << testRaw.numRows() << " test rows." << std::endl;

    // step 1 - prepare the data for training
    auto [trainX, trainYLabels, trainYOneHot] = MultilayerPerceptron::prepare(trainRaw, numCols, numClasses);
    auto [testX, testYLabels, testYOneHot] = MultilayerPerceptron::prepare(testRaw, numCols, numClasses);
    std::cout << "Training and test data separated." << std::endl;
    
    // step 2: create the MLP model
    std::vector<size_t> layerSizes = {numCols}; // input layer: one unit per feature
    layerSizes.insert(layerSizes.end(), hiddenSizes.begin(), hiddenSizes.end());
    layerSizes.push_back(numClasses); // output layer: one unit per class
    MultilayerPerceptron mlp(layerSizes, activations);

    // step 3: train the MLP model
    mlp.trainMLP(trainX, trainYOneHot, lambda, learningRate, maxIterations, batchSize); 

    // step 4: evaluate the MLP model
    double accuracy = mlp.evaluateMLPAccuracy(testX, testYLabels);
    double loss = mlp.evaluateMLPCE(testX, testYOneHot);
    std::cout << "Test loss: " << loss << std::endl;
    std::cout << "Test accuracy: " << accuracy << std::endl;
    return accuracy;
}

int main() {
    try {
        double accuracy = runMultilayerPerceptron("data.csv", /*batchSize=*/4, /*lambda=*/0.01, /*learningRate=*/0.1, /*maxIterations=*/1000, /*tolerance=*/1e-6, /*testRatio=*/0.2);
        std::cout << "Multilayer perceptron completed successfully. Final test accuracy: " << accuracy * 100 << "%" << std::endl;

        std::cout << "\n--- Testing against real-world dataset (Iris: setosa vs versicolor vs virginica) ---\n";
        double irisAccuracy = runMultilayerPerceptron("data_iris_multiclass.csv", /*batchSize=*/16, /*lambda=*/0.01, /*learningRate=*/0.001, /*maxIterations=*/1000, /*tolerance=*/1e-6, /*testRatio=*/0.2);
        std::cout << "Iris dataset multilayer perceptron completed successfully. Final test accuracy: " << irisAccuracy * 100 << "%" << std::endl;

        std::cout << "\n--- Benchmarking against MNIST (10k rows, 784 features, 10 classes) ---\n";
        auto runStart = std::chrono::steady_clock::now();
        double mnistAccuracy = runMultilayerPerceptron("data_mnist.csv", /*batchSize=*/64, /*lambda=*/0.01, /*learningRate=*/0.001, /*maxIterations=*/1000, /*tolerance=*/1e-6, /*testRatio=*/0.2);
        auto runEnd = std::chrono::steady_clock::now();
        std::cout << "[timing] runMultilayerPerceptron: " << std::chrono::duration<double>(runEnd - runStart).count() << "s" << std::endl;
        std::cout << "MNIST multilayer perceptron completed successfully. Final test accuracy: " << mnistAccuracy * 100 << "%" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during multilayer perceptron: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}