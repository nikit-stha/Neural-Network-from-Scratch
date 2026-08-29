#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>
#include <stdexcept>

using namespace std;

namespace nn{
    enum class ActivationType{
        ReLU,
        Sigmoid,
        Softmax,
        Tanh,
        LeakyReLU,
        GeLU,
        Linear
    };

    enum class LossFunction{
        MSELoss,
        BinaryCrossEntropyLoss,
        CrossEntropyLoss,
        MAELoss
    };

    enum class Optimizer{
        SGD,
        Momentum,
        Adam,
        RMSProp
    };

    class DataLoader {
        private:
            vector<vector<double>> data;
            vector<vector<double>> output;
            int batchSize;
            bool shuffle;
            vector<int> indices;
            int currentBatch;
            mt19937 generator;

        public:
            DataLoader(const vector<vector<double>>& data,const vector<vector<double>>& output,int batchSize,bool shuffle = true){
                this->data = data;
                this->output = output;
                this->batchSize = batchSize;
                this->shuffle = shuffle;
                if(data.size() != output.size()){
                    throw invalid_argument(
                        "Data and output must contain the same number of samples."
                    );
                }
                if(batchSize <= 0){
                    throw invalid_argument(
                        "Batch size must be greater than 0."
                    );
                }
                indices.resize(data.size());
                for(int i = 0; i < data.size(); i++){
                    indices[i] = i;
                }
                if(shuffle){
                    this->shuffleData();
                }
            }

            void shuffleData(){
                std::shuffle(indices.begin(),indices.end(),generator);
            }

    };

    struct Parameters{
        vector<vector<double>> &weights;
        vector<double> &bias;

        vector<vector<double>> &dW;
        vector<double> &dB;

        int &inputFeatureDim;
        int &outputFeatureDim;

        ActivationType &activationFunction;
    };

    class Layer{
        private:
            int inputFeatureDim;
            int outputFeatureDim;
            ActivationType activationFunction;

            vector<vector<double>> weights;
            vector<double> bias;

            vector<vector<double>> lastInput;
            vector<vector<double>> lastZ;
            vector<vector<double>> lastOutput;

            vector<vector<double>> dW;
            vector<double> dB;

            void initialize(){
                double limit = sqrt(6.0 / (inputFeatureDim + outputFeatureDim));

                random_device rd;
                mt19937 gen(rd());
                uniform_real_distribution<double> dist(-limit, limit);

                for(int i = 0 ; i < outputFeatureDim ; i++){
                    this->bias[i] = dist(gen);
                    for(int j = 0 ; j < inputFeatureDim ; j++){
                        this->weights[i][j] = dist(gen);
                    }
                }
            }

        public:
            Layer(int inputFeatureDim, int outputFeatureDim, ActivationType activationFunction){
                this->inputFeatureDim = inputFeatureDim;
                this->outputFeatureDim = outputFeatureDim;
                this->activationFunction = activationFunction;

                this->weights.resize((this->outputFeatureDim), vector<double> (this->inputFeatureDim));
                this->bias.resize(this->outputFeatureDim);

                this->dW.resize(this->outputFeatureDim, vector<double>(this->inputFeatureDim, 0.0));
                this->dB.resize(this->outputFeatureDim, 0.0);

                initialize();
            }

            double normalPDF(double x, double mean, double sigma){
            static const double inv_sqrt_2pi = 0.3989422804014327;
            double a = (x - mean) / sigma;
            return (inv_sqrt_2pi / sigma) * exp(-0.5 * a * a);
            }

            vector<double> activation(vector<double> &z){
                vector<double> output(z);
                if(this->activationFunction == ActivationType :: ReLU){
                    for(int i = 0 ; i < z.size() ; i++){
                        output[i] = max(0.0, z[i]);
                    }
                }
                else if(this->activationFunction == ActivationType :: Sigmoid){
                    for(int i = 0 ; i < z.size() ; i++){
                        output[i] = (1.0) / (1.0 + exp(-z[i]));
                    }
                }
                else if(this->activationFunction == ActivationType :: Tanh){
                    for(int i = 0 ; i < z.size() ; i++){
                        output[i] = tanh(z[i]);
                    }
                }
                else if(this->activationFunction == ActivationType::Softmax){
                    double maxZ = *max_element(z.begin(), z.end());
                    double sum = 0.0;

                    for(int i = 0; i < z.size(); i++){
                        sum += exp(z[i] - maxZ);
                    }
                    for(int i = 0; i < z.size(); i++){
                        output[i] = exp(z[i] - maxZ) / sum;
                    }
                }
                else if(this->activationFunction == ActivationType :: GeLU){
                    for(int i = 0 ; i < z.size() ; i++){
                        output[i] = z[i] * normalPDF(z[i], 0.0, 1.0);
                    }
                }
                else if(this->activationFunction == ActivationType :: LeakyReLU){
                    for(int i = 0 ; i < z.size() ; i++){
                        output[i] = max(0.01 * z[i], z[i]);
                    }
                }
                else if(this->activationFunction == ActivationType :: Linear){
                    for(int i = 0 ; i < z.size() ; i++){
                        output[i] = z[i];
                    }
                }
                else{
                    cout << "Incorrect Activation Function" << endl;
                    return {};
                }
                return output;
            }

            vector<double> activationDerivative(vector<double> &z){
                vector<double> derivative(z);
                if(this->activationFunction == ActivationType :: ReLU){
                    for(int i = 0 ; i < z.size() ; i++){
                        if(z[i] <= 0.0){
                            derivative[i] = 0.0;
                        }
                        else{
                            derivative[i] = 1.0;
                        }
                    }
                }
                else if(this->activationFunction == ActivationType :: Sigmoid){
                    vector<double> output = activation(z);
                    for(int i = 0 ; i < z.size() ; i++){
                        derivative[i] = output[i] * (1.0 - output[i]);
                    }
                }
                else if(this->activationFunction == ActivationType :: Tanh){
                    vector<double> output = activation(z);
                    for(int i = 0 ; i < z.size() ; i++){
                        derivative[i] = (1.0 - output[i] * output[i]);
                    }
                }
                else if(this->activationFunction == ActivationType :: GeLU){
                    for(int i = 0 ; i < z.size() ; i++){
                        derivative[i] = normalPDF(z[i], 0.0, 1.0) + z[i] * normalPDF(z[i], 0.0, 1.0);
                    }
                }
                else if(this->activationFunction == ActivationType :: LeakyReLU){
                    for(int i = 0 ; i < z.size() ; i++){
                        if(z[i] <= 0){
                            derivative[i] = 0.01;
                        }
                        else{
                            derivative[i] = 1.0;
                        }
                    }
                }
                else if(this->activationFunction == ActivationType :: Linear){
                    for(int i = 0 ; i < z.size() ; i++){
                        derivative[i] = 1.0;
                    }
                }
                // else if(this->activationFunction == ActivationType :: Softmax){
                //     //! Considering Softmax to be used with CrossCategoricalEntropyLoss
                //     vector<double> softmax = activation(z);
                //     for(int i = 0 ; i < z.size() ; i++){
                //         derivative[i] = softmax[i] - z[i];
                //     }
                // }
                return derivative;
            }

            vector<vector<double>> matmul(const vector<vector<double>> &weights, const vector<vector<double>> &input) {
                    int batchSize = input.size();
                    vector<vector<double>> ans(batchSize, vector<double>(this->outputFeatureDim, 0.0));

                    for (int i = 0; i < batchSize; i++) {
                        for (int j = 0; j < this->outputFeatureDim; j++) {
                            for (int k = 0; k < this->inputFeatureDim; k++) {
                                ans[i][j] += input[i][k] * weights[j][k];
                            }
                        }
                    }
                    return ans;
                }

            vector<vector<double>> forward(vector<vector<double>> &input){
                int batchSize = input.size();

                this->lastInput = input;
                this->lastZ.assign(batchSize, vector<double>(this->outputFeatureDim, 0.0));
                this->lastOutput.assign(batchSize, vector<double>(this->outputFeatureDim, 0.0));

                this->lastZ = matmul(this->weights, input);

                for(int i = 0 ; i < batchSize ; i++){
                    for(int j = 0 ; j < this->lastOutput[0].size() ; j++){
                        this->lastZ[i][j] += this->bias[j];
                    }
                    this->lastOutput[i] = activation(this->lastZ[i]);
                }
                return this->lastOutput;
            }

            vector<vector<double>> backward(vector<vector<double>> &dl_da){
                int batchSize = dl_da.size();
                
                vector<vector<double>> dl_dz(batchSize, vector<double>(this->outputFeatureDim));

                if (this->activationFunction == ActivationType::Softmax) {
                    //! Only For SoftMax (Bug)!
                    for(int i = 0; i < batchSize; i++){
                        for(int j = 0; j < this->outputFeatureDim; j++){
                            dl_dz[i][j] = dl_da[i][j];
                        }
                    }
                }
                else{
                    for(int i = 0 ; i < batchSize ; i++){
                        vector<double> rows = activationDerivative(this->lastZ[i]);
                        for(int j = 0 ; j < this->outputFeatureDim ; j++){
                            dl_dz[i][j] = dl_da[i][j] * rows[j];
                        }
                    }
                }
                for(int j = 0; j < outputFeatureDim; j++){
                    for(int k = 0; k < inputFeatureDim; k++){
                        double sum = 0.0;
                        for (int i = 0; i < batchSize; i++) {
                            sum += dl_dz[i][j] * lastInput[i][k];
                        }
                        this->dW[j][k] = sum / batchSize;
                    }
                }

                for(int j = 0; j < outputFeatureDim; j++){
                    double sum = 0.0;
                    for(int i = 0; i < batchSize; i++){
                    sum += dl_dz[i][j];
                    }
                    this->dB[j] = sum / batchSize;
                }
                vector<vector<double>> dA_prev(batchSize, vector<double>(inputFeatureDim, 0.0));
                for (int i = 0; i < batchSize; i++) {
                    for (int k = 0; k < inputFeatureDim; k++) {
                        for (int j = 0; j < outputFeatureDim; j++) {
                            dA_prev[i][k] += dl_dz[i][j] * weights[j][k];
                        }
                    }
                }
                return dA_prev;
            }

            Parameters getParameters(){
                return{
                    this->weights,
                    this->bias,
                    this->dW,
                    this->dB,
                    this->inputFeatureDim,
                    this->outputFeatureDim,
                    this->activationFunction,
                };
            }
        
            vector<vector<double>> &getWeights(){
                return this->weights;
            }

            vector<double> &getBias(){
                return this->bias;
            }
    };

    class Network{
        private:
            vector<Layer> layers;
            double learningRate;

            LossFunction lossFunction;
            Optimizer optimizer;

        public:
            Network(LossFunction lossFunction, Optimizer optimizer, double learningRate = 0.01){
                this->learningRate = learningRate;
                this->lossFunction = lossFunction;
                this->optimizer = optimizer;
            }
            
            void addLayer(int inputFeatureDim, int outputFeatureDim, ActivationType activationType){
                layers.emplace_back(inputFeatureDim, outputFeatureDim, activationType);
            }

            vector<vector<double>> forwardpass(vector<vector<double>> &input){
                vector<vector<double>> ans = input;
                for(auto &layer : layers){
                    ans = layer.forward(ans);
                }
                return ans;
            }

            double loss(vector<vector<double>> &input, vector<vector<double>> &output){
                int batchSize = input.size();
                if(batchSize != output.size()){
                    cout << "Output and Input Dimention Mismatched" << endl;
                    return 0;
                }
                double loss = 0.0;

                if(this->lossFunction == LossFunction :: MSELoss){
                    double sum = 0.0;
                    vector<vector<double>> prediction = forwardpass(input);

                    for(int i = 0 ; i < batchSize ; i++){
                        for(int j = 0 ; j < prediction[0].size() ; j++){
                            sum += (double) ((1.0 / 2.0) * (prediction[i][j] - output[i][j]) * (prediction[i][j] - output[i][j]));
                        }
                    }
                    return (double)(sum / batchSize);
                }
                else if(this->lossFunction == LossFunction :: CrossEntropyLoss){
                    double sum = 0.0;
                    vector<vector<double>> prediction = forwardpass(input);
                    for(int i = 0 ; i < batchSize ; i++){
                        for(int j = 0 ; j < prediction[0].size() ; j++){
                            double p = max(prediction[i][j], 1e-15);
                            sum -= output[i][j] * log(p);
                        }
                    }
                    return (double) (sum / batchSize);
                }
                else if(this->lossFunction == LossFunction :: MAELoss){
                    double sum = 0;
                    vector<vector<double>> prediction = forwardpass(input);

                    for(int i = 0 ; i < batchSize ; i++){
                        for(int j = 0 ; j < prediction[0].size() ; j++){
                            sum += (double) abs((prediction[i][j] - output[i][j]));
                        }
                    }
                    return (double)(sum / batchSize);  
                }
                else if(this->lossFunction == LossFunction :: BinaryCrossEntropyLoss){
                    //! Binary Cross Entropy Loss Not Completed
                    int sum = 0;
                    return 0.0;
                }
                else{
                    cout << "Invalid Loss Function" << endl;
                    return 0.0;
                }
            }

            void backwardPass(vector<vector<double>> &input, vector<vector<double>> &target) {

                vector<vector<double>> y_pred = forwardpass(input);
                int bSize = input.size();
                int outDim = target[0].size();

                vector<vector<double>> dl_da(bSize, vector<double>(outDim, 0.0));

                if(this->lossFunction == LossFunction :: MSELoss){
                    for(int i = 0; i < bSize; i++){
                        for(int j = 0; j < outDim; j++){
                            dl_da[i][j] = y_pred[i][j] - target[i][j];
                        }
                    }
                } 
                else if(this->lossFunction == LossFunction :: MAELoss){
                    for(int i = 0; i < bSize; i++){
                        for(int j = 0; j < outDim; j++){
                            if (y_pred[i][j] > target[i][j]) dl_da[i][j] = 1.0;
                            else if (y_pred[i][j] < target[i][j]) dl_da[i][j] = -1.0;
                            else dl_da[i][j] = 0.0;
                        }
                    }
                }

                else if(this->lossFunction == LossFunction :: CrossEntropyLoss){
                    for(int i = 0 ; i < bSize ; i++){
                        for(int j = 0 ; j < outDim ; j++){
                            dl_da[i][j] = y_pred[i][j] - target[i][j];
                        }
                    }
                }

                //! Other Loss Function is not Complete

                vector<vector<double>> currentGradient = dl_da;
                if (this->optimizer == Optimizer::SGD) {
                    for (int l = (int)layers.size() - 1; l >= 0; l--) {
                        currentGradient = layers[l].backward(currentGradient);
                        Parameters param = layers[l].getParameters();
                        for (int i = 0; i < param.outputFeatureDim; i++) {
                            param.bias[i] -= this->learningRate * param.dB[i];

                            for (int j = 0; j < param.inputFeatureDim; j++) {
                                param.weights[i][j] -= this->learningRate * param.dW[i][j];
                            }
                        }
                    }
                }
                else if(this->optimizer == Optimizer :: Momentum){
                    for(int l = layers.size() - 1 ; l >= 0 ; l--){
                        Parameters param = layers[l].getParameters();
                        currentGradient = layers[l].backward(currentGradient);
                        for(int i = 0 ; i < param.outputFeatureDim ; i++){
                            for(int j = 0 ; j < param.inputFeatureDim ; i++){
                                //! Not Complete
                            }
                        }
                    }
                }
                else{
                    cout << "Invalid Optimizer " << endl;
                    return ;
                }
            }
            
            void saveModel(const string &fileName){

                //Create File
                ofstream file(fileName, ios::binary);
                if(!file){
                    throw runtime_error{"Couldnot open file for writing"};
                }

                // Set Magic Number
                const char magic[] = "Network";
                file.write(magic, sizeof(magic));

                // Write Number of Layers
                int numLayers = layers.size();
                file.write((char *)(&numLayers), sizeof(numLayers));

                //Save Every Layer
                for(auto &layer : layers){
                    Parameters params = layer.getParameters();

                    // Write Input Feature Dimension
                    file.write((char *)(&params.inputFeatureDim), sizeof(params.inputFeatureDim));

                    // Write Output Feature Dimension
                    file.write((char *)(&params.outputFeatureDim), sizeof(params.outputFeatureDim));

                    // Write Activation Function
                    file.write((char *)(&params.activationFunction), sizeof(params.activationFunction));
                    
                    // Write Weights
                    for(int i = 0 ; i < params.outputFeatureDim ; i++){
                        file.write((char *)(params.weights[i].data()), params.inputFeatureDim * sizeof(double));
                    }

                    // Write Bias
                    file.write((char *)(params.bias.data()), params.outputFeatureDim * sizeof(double));
                }
                    // Close File
                    file.close();
            }

            void loadModel(const string &fileName){
                // Open File
                ifstream file(fileName, ios::binary);
                if(!file){
                    throw runtime_error {"Could not open model file for reading"};
                }

                // Check Magic number
                char magic[8];
                file.read(magic, sizeof(magic));

                if(string(magic) != "Network"){
                    throw runtime_error{"Invalid model file"};
                }

                // Number of Layers
                int numLayers;
                file.read((char *)(&numLayers), sizeof(numLayers));

                //Remove Old Architecture
                layers.clear();

                // Update Layer
                for(int i = 0 ; i < numLayers ; i++){
                    int outputFeatureDim;
                    int inputFeatureDim;
                    ActivationType activationType;

                    // Read Input Size
                    file.read((char *)(&inputFeatureDim), sizeof(inputFeatureDim));

                    // Read Output Size
                    file.read((char *)(&outputFeatureDim), sizeof(outputFeatureDim));

                    // Read Activation Function
                    file.read((char *)(&activationType), sizeof(ActivationType));

                    // Create Layer
                    Layer layerX(inputFeatureDim, outputFeatureDim, activationType);

                    // Load Weights
                    for(int i = 0 ; i < outputFeatureDim; i++){
                        file.read((char *)(layerX.getWeights()[i].data()), inputFeatureDim * sizeof(double));
                    }

                    // Load Bias
                    file.read((char *)(layerX.getBias().data()), outputFeatureDim * sizeof(double));

                    layers.push_back(move(layerX));
                }
                file.close();
            }
    };
};
