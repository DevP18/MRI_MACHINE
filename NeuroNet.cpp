#include "NeuroNet.h"
#include <cmath>
#include <fstream>
#include <cassert>
using namespace std;

double Neuron::eta = 0.01;   // Start lower as you did in main
double Neuron::alpha = 0.5;  // Keep the higher momentum
double net::m_recentAverageSmoothingFactor = 40.0;

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) : m_myIndex(myIndex), m_outputVal(0.0), m_gradient(0.0) {
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back({ randomWeight(), 0.0 });
    }
}

// He initialization: scale by sqrt(2/fan_in) so weighted sums stay ~[-1,1]
// Called after construction once we know the layer size
void net::initWeightsHe() {
    for (unsigned l = 0; l < m_layers.size() - 1; ++l) {
        unsigned fan_in = m_layers[l].size(); // includes bias neuron
        double scale = sqrt(2.0 / fan_in);
        for (unsigned n = 0; n < m_layers[l].size(); ++n) {
            for (auto &c : m_layers[l][n].getOutputWeightsReference()) {
                c.weight *= scale;
            }
        }
    }
}

// Leaky ReLU for hidden layers
double Neuron::transferFunction(double x) {
    return x > 0 ? x : 0.01 * x;
}

double Neuron::transferFunctionDerivative(double x) {
    return x > 0 ? 1.0 : 0.01;
}

// Used by Neuron's own feedForward (hidden layers only, called from old code path)
void Neuron::feedForward(const Layer &prevLayer) {
    double sum = 0.0;
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        // FIX: multiply output value by weight
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    m_outputVal = Neuron::transferFunction(sum);
}

// FIX: Output gradient for sigmoid output: delta * sigmoid_derivative
// sigmoid'(x) = output * (1 - output)  — no need to call transferFunctionDerivative
void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    // Sigmoid derivative: out*(1-out)
    m_gradient = delta * m_outputVal * (1.0 - m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer) {
    double weightDecay = 0.00001;  // back to original
    // Clip gradient to prevent exploding updates
    double clippedGradient = m_gradient;
    if (clippedGradient > 1.0)  clippedGradient = 1.0;
    if (clippedGradient < -1.0) clippedGradient = -1.0;

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
            eta * neuron.getOutputVal() * clippedGradient
            + alpha * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight =
            neuron.m_outputWeights[m_myIndex].weight * (1.0 - weightDecay) + newDeltaWeight;
    }
}

// --- NET CLASS METHODS ---

net::net(const std::vector<unsigned> &topology) : m_error(0.0), m_recentAverageError(0.0) {
    unsigned numLayers = topology.size();
    for (unsigned l = 0; l < numLayers; ++l) {
        m_layers.push_back(Layer());
        unsigned numOutputs = (l == topology.size() - 1) ? 0 : topology[l + 1];
        for (unsigned n = 0; n <= topology[l]; ++n) {
            m_layers.back().push_back(Neuron(numOutputs, n));
        }
        m_layers.back().back().setOutputVal(1.0); // Bias neuron
    }
    initWeightsHe(); // Scale all weights so sums don't explode with large fan_in
}

void net::feedForward(const vector<double> &inputVals) {
    // 1. Set Input Layer
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // 2. Forward pass through hidden + output layers
    for (unsigned l = 1; l < m_layers.size(); ++l) {
        Layer &prevLayer = m_layers[l - 1];
        for (unsigned n = 0; n < m_layers[l].size() - 1; ++n) {

            // FIX: correctly compute weighted sum (was missing getOutputVal() multiply)
            double sum = 0.0;
            for (unsigned p = 0; p < prevLayer.size(); ++p) {
                sum += prevLayer[p].getOutputVal() * prevLayer[p].getOutputWeightsReference()[n].weight;
            }

            if (l == m_layers.size() - 1) {
                // OUTPUT LAYER: Sigmoid → probability in [0, 1]
                m_layers[l][n].setOutputVal(1.0 / (1.0 + exp(-sum)));
            } else {
                // HIDDEN LAYERS: Leaky ReLU
                m_layers[l][n].setOutputVal(sum > 0 ? sum : 0.01 * sum);
            }
        }
    }
}

void net::backprop(const std::vector<double> &targetVals) {
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);

    m_recentAverageError =
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
        / (m_recentAverageSmoothingFactor + 1.0);

    // Output layer gradients (uses sigmoid-aware formula)
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Hidden layer gradients
    for (unsigned l = m_layers.size() - 2; l > 0; --l) {
        Layer &hiddenLayer = m_layers[l];
        Layer &nextLayer = m_layers[l + 1];
        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // Update weights
    for (unsigned l = m_layers.size() - 1; l > 0; --l) {
        Layer &layer = m_layers[l];
        Layer &prevLayer = m_layers[l - 1];
        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void net::getresults(std::vector<double> &resultVals) const {
    resultVals.clear();
    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void net::saveWeights(const std::string &filename) const {
    std::ofstream fout(filename, std::ios::out | std::ios::trunc);
    if (!fout.is_open()) {
        std::cerr << "CRITICAL ERROR: Could not open " << filename << " for writing!" << std::endl;
        return;
    }
    for (unsigned l = 0; l < m_layers.size() - 1; ++l) {
        for (unsigned n = 0; n < m_layers[l].size(); ++n) {
            const std::vector<Connections> &conns =
                const_cast<Layer&>(m_layers[l])[n].getOutputWeightsReference();
            for (auto &c : conns) {
                fout << c.weight << " ";
            }
            fout << "\n";
        }
    }
    fout.flush();
    fout.close();
    std::cout << ">>> Weights saved to " << filename << " <<<" << std::endl;
}

bool net::loadWeights(const std::string &filename) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "ERROR: Could not open " << filename << " for reading!" << std::endl;
        return false; // Tells the if-statement in main that it failed
    }

    for (unsigned l = 0; l < m_layers.size() - 1; ++l) {
        for (unsigned n = 0; n < m_layers[l].size(); ++n) {
            std::vector<Connections> &conns = m_layers[l][n].getOutputWeightsReference();
            for (auto &c : conns) fin >> c.weight;
        }
    }

    fin.close();
    std::cout << ">>> Weights loaded successfully <<<" << std::endl;
    return true; // Tells the if-statement in main it worked!
}