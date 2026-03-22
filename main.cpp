#include "NeuroNet.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <iomanip>

using namespace std;

double trainEpoch(net &myNet, string filename, int epochNum) {
    ifstream dataFile(filename);
    if (!dataFile.is_open()) return 1.0;

    string line;
    vector<string> allLines;
    while (getline(dataFile, line)) if (!line.empty()) allLines.push_back(line);
    dataFile.close();

    static mt19937 g(time(0));
    shuffle(allLines.begin(), allLines.end(), g);

    double totalError = 0;
    int trainedCount = 0;
    int totalLines = allLines.size();

    for (const string &l : allLines) {
        stringstream ss(l);
        double target;
        if (!(ss >> target)) continue;

        vector<double> inputs;
        double pixel;
        while (ss >> pixel) inputs.push_back(pixel);

        if (inputs.size() == 1024) {
            myNet.feedForward(inputs);
            myNet.backprop({target});

            // Accumulate raw per-sample squared error, not the smoothed value
            vector<double> result;
            myNet.getresults(result);
            double delta = target - result[0];
            totalError += delta * delta;
            trainedCount++;

            if (trainedCount % 50 == 0) {
                cout << "\rEpoch [" << epochNum << "] Progress: "
                     << (trainedCount * 100 / totalLines) << "% " << flush;
            }
        }
    }

    if (trainedCount > 0) {
        double avgErr = sqrt(totalError / trainedCount);
        cout << "\rEpoch [" << epochNum << "] Samples: " << trainedCount
             << " | Avg Error: " << fixed << setprecision(6) << avgErr << endl;

        ofstream log("training_log.txt", ios::app);
        log << epochNum << " " << avgErr << endl;

        return avgErr;
    }
    return 1.0;
}

int main() {
    vector<unsigned> topology = {1024, 64, 16, 1};
    net myNet(topology);

    Neuron::alpha = 0.5;
    Neuron::eta = 0.01;

    cout << "--- STARTING RECOVERY TRAINING ---" << endl;

    double bestError = 1e9;
    for (int i = 1; i <= 50; ++i) {
        double err = trainEpoch(myNet, "mri_train.txt", i);
        if (err < bestError) {
            bestError = err;
            myNet.saveWeights("mri_best.weights");
            cout << "  ^ New best (" << fixed << setprecision(6) << bestError << ") saved." << endl;
        }
    }

    cout << "\nLoading best weights..." << endl;
    myNet.loadWeights("mri_best.weights");

    cout << "\n--- STARTING TEST & AUDIT EXPORT ---" << endl;
    ifstream testFile("mri_test.txt");
    ifstream metaFile("mri_test_meta.txt");
    ofstream auditFile("test_results.txt");

    int correct = 0, total = 0, fp = 0, fn = 0;
    string line, meta;

    while (getline(testFile, line) && getline(metaFile, meta)) {
        stringstream ss(line);
        double target; ss >> target;
        vector<double> inputs;
        double p; while (ss >> p) inputs.push_back(p);

        if (inputs.size() == 1024) {
            myNet.feedForward(inputs);
            vector<double> results;
            myNet.getresults(results);

            double rawScore = results[0];
            int prediction = (rawScore > 0.5) ? 1 : 0;

            auditFile << meta << " " << target << " " << rawScore << endl;

            if (prediction == (int)target) correct++;
            else (prediction == 1) ? fp++ : fn++;
            total++;
        }
    }
    auditFile.close();

    cout << "Final Accuracy: " << (double)correct / total * 100.0 << "%" << endl;
    cout << "Missed (FN): " << fn << " | False Alarms (FP): " << fp << endl;

    return 0;
}