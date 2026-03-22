#ifndef NEURONET_H
#define NEURONET_H

#include <vector>
#include <string>
#include <iostream>

struct Connections {
    double weight;
    double deltaWeight;
};

class Neuron;
typedef std::vector<Neuron> Layer;

class Neuron {
public:
    static double eta;   // Learning rate [0.0...1.0]
    static double alpha;  // Momentum [0.0...n]
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal() const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

    std::vector<Connections> &getOutputWeightsReference() { return m_outputWeights; }

private:
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    // FIX: weights in [-1, 1] range for better initialization
    static double randomWeight(void) { return (rand() / double(RAND_MAX)) * 2.0 - 1.0; }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    std::vector<Connections> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

class net {
public:
    net(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backprop(const std::vector<double> &targetVals);
    void getresults(std::vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }

    void saveWeights(const std::string &filename) const;
    bool loadWeights(const std::string &filename);

private:
    std::vector<Layer> m_layers;
    void initWeightsHe(); // Scale weights by sqrt(2/fan_in) after construction
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};

#endif