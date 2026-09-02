#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>
#include <cmath>

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
            std::vector<std::vector<double>> data;
            std::vector<std::vector<double>> output;
            int batchSize;
            bool shuffle;
            std::vector<int> indices;
            int currentBatch;
            std::mt19937 generator;

        public:
            DataLoader(const std::vector<std::vector<double>>& data,const std::vector<std::vector<double>>& output,int batchSize,bool shuffle = true){
                this->data = data;
                this->output = output;
                this->batchSize = batchSize;
                this->shuffle = shuffle;
                if(data.size() != output.size()){
                    throw std::invalid_argument(
                        "Data and output must contain the same number of samples."
                    );
                }
                if(batchSize <= 0){
                    throw std::invalid_argument(
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
        std::vector<std::vector<double>> &weights;
        std::vector<double> &bias;

        std::vector<std::vector<double>> &dW;
        std::vector<double> &dB;

        int &inputFeatureDim;
        int &outputFeatureDim;

        ActivationType &activationType;

        std::vector<std::vector<double>> &velocitydW;
        std::vector<double> &velocitydB;
        std::vector<std::vector<double>> &adamSecondMomentdW;
        std::vector<double> &adamSecondMomentdB;
        int &adamStep;
    };

    class Activation{
        private:
            ActivationType activationType;
        public:
            Activation(ActivationType activationType){
                this->activationType = activationType;
            }
            
            std::vector<double> activation(std::vector<double> &z){
                std::vector<double> output(z);
                if(this->activationType == ActivationType :: ReLU){
                    for(int i = 0 ; i < z.size() ; i++){
                        output[i] = std::max(0.0, z[i]);
                    }
                }
                else if(this->activationType == ActivationType :: Sigmoid){
                    for(int i = 0 ; i < z.size() ; i++){
                        output[i] = (1.0) / (1.0 + exp(-z[i]));
                    }
                }
                else if(this->activationType == ActivationType :: Tanh){
                    for(int i = 0 ; i < z.size() ; i++){
                        output[i] = tanh(z[i]);
                    }
                }
                else if(this->activationType == ActivationType::Softmax){
                    double maxZ = *max_element(z.begin(), z.end());
                    double sum = 0.0;

                    for(int i = 0; i < z.size(); i++){
                        sum += exp(z[i] - maxZ);
                    }
                    for(int i = 0; i < z.size(); i++){
                        output[i] = exp(z[i] - maxZ) / sum;
                    }
                }
                else if(this->activationType == ActivationType :: GeLU){
                    for(int i = 0 ; i < z.size() ; i++){
                        output[i] = 0.5 * z[i] * (1.0 + std::erf(z[i]/std::sqrt(2.0)));
                    }
                }
                else if(this->activationType == ActivationType :: LeakyReLU){
                    for(int i = 0 ; i < z.size() ; i++){
                        output[i] = std::max(0.01 * z[i], z[i]);
                    }
                }
                else if(this->activationType == ActivationType :: Linear){
                    for(int i = 0 ; i < z.size() ; i++){
                        output[i] = z[i];
                    }
                }
                else{
                    throw std::invalid_argument{"Activation Function is Invalid"};
                }
                return output;
            }

            std::vector<double> activationDerivative(std::vector<double> &z){
                std::vector<double> derivative(z);
                if(this->activationType == ActivationType :: ReLU){
                    for(int i = 0 ; i < z.size() ; i++){
                        if(z[i] <= 0.0){
                            derivative[i] = 0.0;
                        }
                        else{
                            derivative[i] = 1.0;
                        }
                    }
                }
                else if(this->activationType == ActivationType :: Sigmoid){
                    std::vector<double> output = activation(z);
                    for(int i = 0 ; i < z.size() ; i++){
                        derivative[i] = output[i] * (1.0 - output[i]);
                    }
                }
                else if(this->activationType == ActivationType :: Tanh){
                    std::vector<double> output = activation(z);
                    for(int i = 0 ; i < z.size() ; i++){
                        derivative[i] = (1.0 - output[i] * output[i]);
                    }
                }
                else if(this->activationType == ActivationType :: GeLU){
                    for(int i = 0 ; i < z.size() ; i++){
                        double cdf = 0.5 * (1.0 + std::erf(z[i] / std::sqrt(2.0)));
                        double pdf = 0.3989422804014327 * std::exp(-0.5 * z[i] * z[i]);
                        derivative[i] = cdf + z[i] * pdf;
                    }
                }
                else if(this->activationType == ActivationType :: LeakyReLU){
                    for(int i = 0 ; i < z.size() ; i++){
                        if(z[i] <= 0){
                            derivative[i] = 0.01;
                        }
                        else{
                            derivative[i] = 1.0;
                        }
                    }
                }
                else if(this->activationType == ActivationType :: Softmax){
                    std::vector<double> output = activation(z);
                    for(int i = 0 ; i < z.size() ; i++){
                        derivative[i] = output[i] * (1.0 - output[i]);
                    }
                }
                else if(this->activationType == ActivationType :: Linear){
                    for(int i = 0 ; i < z.size() ; i++){
                        derivative[i] = 1.0;
                    }
                }
                return derivative;
            }
    
        };

    class Layer{
        private:
            int inputFeatureDim;
            int outputFeatureDim;
            ActivationType activationType;
            Activation activationFunction;
            

            std::vector<std::vector<double>> weights;
            std::vector<double> bias;

            std::vector<std::vector<double>> lastInput;
            std::vector<std::vector<double>> lastZ;
            std::vector<std::vector<double>> lastOutput;

            std::vector<std::vector<double>> dW;
            std::vector<double> dB;

            std::vector<std::vector<double>>velocitydW;
            std::vector<double>velocitydB;
            std::vector<std::vector<double>>adamSecondMomentdW;
            std::vector<double>adamSecondMomentdB;
            int adamStep;

            void initialize(){
                double limit = sqrt(6.0 / (inputFeatureDim + outputFeatureDim));

                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<double> dist(-limit, limit);

                for(int i = 0 ; i < outputFeatureDim ; i++){
                    this->bias[i] = dist(gen);
                    for(int j = 0 ; j < inputFeatureDim ; j++){
                        this->weights[i][j] = dist(gen);
                    }
                }
            }

        public:
            Layer(int inputFeatureDim, int outputFeatureDim, ActivationType activationType) : activationFunction(activationType){
                this->inputFeatureDim = inputFeatureDim;
                this->outputFeatureDim = outputFeatureDim;
                this->activationType = activationType;
                this->adamStep = 0;

                this->weights.resize((this->outputFeatureDim), std::vector<double> (this->inputFeatureDim));
                this->bias.resize(this->outputFeatureDim);

                this->dW.resize(this->outputFeatureDim, std::vector<double>(this->inputFeatureDim, 0.0));
                this->dB.resize(this->outputFeatureDim, 0.0);

                this->velocitydW.resize(this->outputFeatureDim, std::vector<double>(this->inputFeatureDim, 0.0));
                this->velocitydB.resize(this->outputFeatureDim, 0.0);
                this->adamSecondMomentdW.resize(this->outputFeatureDim, std::vector<double>(this->inputFeatureDim, 0.0));
                this->adamSecondMomentdB.resize(this->outputFeatureDim, 0.0);

                initialize();
            }

            std::vector<std::vector<double>> matmul(const std::vector<std::vector<double>> &weights, const std::vector<std::vector<double>> &input) {
                    int batchSize = input.size();
                    std::vector<std::vector<double>> ans(batchSize, std::vector<double>(this->outputFeatureDim, 0.0));

                    for (int i = 0; i < batchSize; i++) {
                        for (int j = 0; j < this->outputFeatureDim; j++) {
                            for (int k = 0; k < this->inputFeatureDim; k++) {
                                ans[i][j] += input[i][k] * weights[j][k];
                            }
                        }
                    }
                    return ans;
                }

            std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &input){
                int batchSize = input.size();

                this->lastInput = input;
                this->lastZ.assign(batchSize, std::vector<double>(this->outputFeatureDim, 0.0));
                this->lastOutput.assign(batchSize, std::vector<double>(this->outputFeatureDim, 0.0));

                this->lastZ = matmul(this->weights, input);

                for(int i = 0 ; i < batchSize ; i++){
                    for(int j = 0 ; j < this->lastOutput[0].size() ; j++){
                        this->lastZ[i][j] += this->bias[j];
                    }
                    this->lastOutput[i] = activationFunction.activation(this->lastZ[i]);
                }
                return this->lastOutput;
            }

            std::vector<std::vector<double>> backward(std::vector<std::vector<double>> &dl_da){
                int batchSize = dl_da.size();
                
                std::vector<std::vector<double>> dl_dz(batchSize, std::vector<double>(this->outputFeatureDim));

                if (this->activationType == ActivationType::Softmax) {
                    //! Only For SoftMax (Bug)!
                    for(int i = 0; i < batchSize; i++){
                        for(int j = 0; j < this->outputFeatureDim; j++){
                            dl_dz[i][j] = dl_da[i][j];
                        }
                    }
                }
                else{
                    for(int i = 0 ; i < batchSize ; i++){
                        std::vector<double> rows = activationFunction.activationDerivative(this->lastZ[i]);
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
                std::vector<std::vector<double>> dA_prev(batchSize, std::vector<double>(inputFeatureDim, 0.0));
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
                    this->activationType,

                    this->velocitydW,
                    this->velocitydB,

                    this->adamSecondMomentdW,
                    this->adamSecondMomentdB,
                    this->adamStep
                };
            }
        
            std::vector<std::vector<double>> &getWeights(){
                return this->weights;
            }

            std::vector<double> &getBias(){
                return this->bias;
            }
    };

    class Conv2DLayer{
        private:
            int inputChannel;
            int outputChannel;
            int kernel_size;
            int stride;
            ActivationType activationType;
            Activation activationFunction;

            std::vector<std::vector<std::vector<std::vector<double>>>> weights; // [outputChannel][inputChannel][height][width]
            std::vector<std::vector<double>> bias;

            std::vector<std::vector<std::vector<double>>> lastInput;
            std::vector<std::vector<std::vector<double>>> lastZ;
            std::vector<std::vector<std::vector<double>>> lastOutput;

            void initialze(){
                int outputChannel = this->weights.size();
                int inputChannel = this->weights[0].size();
                int numRows = this->weights[0][0].size();
                int numCols = this->weights[0][0][0].size();

                int kernelSpacialSize = numRows * numCols;
                int fanIn = kernelSpacialSize * inputChannel;
                int fanOut = kernelSpacialSize * outputChannel;

                double limit = sqrt(6.0 / (fanIn + fanOut));

                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<double> dist(-limit, limit);

                for(int oc = 0 ; oc < outputChannel ; oc++){
                    for(int ic = 0 ; ic < inputChannel ; ic++){
                        for(int i = 0 ; i < numRows ; i++){
                            for(int j = 0 ; j < numCols ; j++){
                                this->weights[oc][ic][i][j] = dist(gen);
                            }
                        }
                    }
                }

            }
        
        public:

            Conv2DLayer(int inputChannel, int outputChannel, int kernel_size, int stride, ActivationType activationType) : activationFunction(activationType){
                this->inputChannel = inputChannel;
                this->outputChannel = outputChannel;
                this->activationType = activationType;
                this->kernel_size = kernel_size;
                this->stride = stride;

                this->weights.resize(this->outputChannel, std::vector<std::vector<std::vector<double>>>(this->inputChannel, std::vector<std::vector<double>>(kernel_size, std::vector<double>(kernel_size, 0.0))));
                //! Skip Bias for Now

                initialze();
            }
            
            std::vector<std::vector<std::vector<double>>> convolve(std::vector<std::vector<std::vector<double>>> &input, std::vector<std::vector<std::vector<std::vector<double>>>> &kernel){
                int inputChannel = input.size();
                int outputChannel = kernel.size();
                
                int inputRows = input[0].size();
                int inputCols = input[0][0].size();

                int kernelRows = kernel[0][0].size();
                int kernelCols = kernel[0][0][0].size();

                int outputRows = (inputRows - kernelRows) / stride + 1;
                int outputCols = (inputCols - kernelCols) / stride + 1;

                std::vector<std::vector<std::vector<double>>> output(outputChannel, std::vector<std::vector<double>>(outputRows, std::vector<double>(outputCols, 0.0)));

                // For every output channel
                for(int oc = 0 ; oc < outputChannel ; oc++){

                    // Every output pixel
                    for(int i = 0 ; i < outputRows ; i++){
                        for(int j = 0 ; j < outputCols ; j++){
                            double sum = 0;
                            // Sum over every input channel
                            for(int ic = 0 ; ic < inputChannel ; ic++){
                                // Kernel
                                for(int k = 0 ; k < kernelRows ; k++){
                                    for(int l = 0 ; l < kernelCols ; l++){
                                        sum += input[ic][i * stride + k][j * stride + l] * kernel[oc][ic][k][l];
                                    }
                                }
                            }
                            output[oc][i][j] = sum;
                        }
                    }
                }
                return output;
            }

            std::vector<std::vector<std::vector<double>>> forward(std::vector<std::vector<std::vector<double>>> &input){
                int numChannels = input.size();

                int numRows = input[0].size();
                int numCols = input[0][0].size();

                int kernelRows = this->weights[0][0].size();
                int kernelCols = this->weights[0][0][0].size();

                int outputRows = (numRows - kernelRows) / this->stride + 1;
                int outputCols = (numCols - kernelCols) / this->stride + 1;

                std::vector<std::vector<std::vector<double>>> z = convolve(input, weights);
                std::vector<std::vector<std::vector<double>>> output = z;

                int outputChannels = z.size();
                for(int oc = 0 ; oc < outputChannels ; oc++){
                    for(int i = 0 ; i < outputRows ; i++){
                        output[oc][i] = activationFunction.activation(z[oc][i]);
                    }
                }
                return output;
            }

            //! Not Complete
            std::vector<std::vector<std::vector<double>>> backward(){

            }
};

    class Network{
        private:
            std::vector<Layer> layers;
            double learningRate;

            LossFunction lossFunction;
            Optimizer optimizer;

        public:
            Network(LossFunction lossFunction, Optimizer optimizer, double learningRate = 0.01){
                this->learningRate = learningRate;
                this->lossFunction = lossFunction;
                this->optimizer = optimizer;
            }
            
            void Linear(int inputFeatureDim, int outputFeatureDim, ActivationType activationType){
                layers.emplace_back(inputFeatureDim, outputFeatureDim, activationType);
            }

            std::vector<std::vector<double>> forwardpass(std::vector<std::vector<double>> &input){
                std::vector<std::vector<double>> ans = input;
                for(auto &layer : layers){
                    ans = layer.forward(ans);
                }
                return ans;
            }

            double loss(std::vector<std::vector<double>> &input, std::vector<std::vector<double>> &output){
                int batchSize = input.size();
                if(batchSize != output.size()){
                    throw std::invalid_argument{"Input and Output Dimension Mismatched"};
                }
              double loss = 0.0;

                if(this->lossFunction == LossFunction :: MSELoss){
                    double sum = 0.0;
                    std::vector<std::vector<double>> prediction = forwardpass(input);

                    for(int i = 0 ; i < batchSize ; i++){
                        for(int j = 0 ; j < prediction[0].size() ; j++){
                            sum += (double) ((1.0 / 2.0) * (prediction[i][j] - output[i][j]) * (prediction[i][j] - output[i][j]));
                        }
                    }
                    return (double)(sum / batchSize);
                }
                else if(this->lossFunction == LossFunction :: CrossEntropyLoss){
                    double sum = 0.0;
                    std::vector<std::vector<double>> prediction = forwardpass(input);
                    for(int i = 0 ; i < batchSize ; i++){
                        for(int j = 0 ; j < prediction[0].size() ; j++){
                            double p = std::max(prediction[i][j], 1e-15);
                            sum -= output[i][j] * log(p);
                        }
                    }
                    return (double) (sum / batchSize);
                }
                else if(this->lossFunction == LossFunction :: MAELoss){
                    double sum = 0;
                    std::vector<std::vector<double>> prediction = forwardpass(input);

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
                    throw std::invalid_argument{"Invalid Loss Function"};
                }
            }

            void backwardPass(std::vector<std::vector<double>> &input, std::vector<std::vector<double>> &target) {

                std::vector<std::vector<double>> y_pred = forwardpass(input);
                int bSize = input.size();
                int outDim = target[0].size();

                std::vector<std::vector<double>> dl_da(bSize, std::vector<double>(outDim, 0.0));

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

                std::vector<std::vector<double>> currentGradient = dl_da;
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
                    const double beta = 0.9;
                    for(int l = layers.size() - 1 ; l >= 0 ; l--){
                        currentGradient = layers[l].backward(currentGradient);
                        Parameters param = layers[l].getParameters();

                        for(int i = 0 ; i < param.outputFeatureDim ; i++){
                            param.velocitydB[i] = beta * param.velocitydB[i] + (1.0 - beta) * param.dB[i];
                            param.bias[i] -= this->learningRate * param.velocitydB[i];

                            for(int j = 0 ; j < param.inputFeatureDim ; j++){
                                param.velocitydW[i][j] = beta * param.velocitydW[i][j] + (1.0 - beta) * param.dW[i][j];
                                param.weights[i][j] -= this->learningRate * param.velocitydW[i][j];
                            }
                        }
                    }
                }
                else if(this->optimizer == Optimizer :: Adam){
                    const double beta1 = 0.9;
                    const double beta2 = 0.999;
                    const double epsilon = 1e-8;

                    for(int l = layers.size() - 1 ; l >= 0 ; l--){
                        currentGradient = layers[l].backward(currentGradient);
                        Parameters param = layers[l].getParameters();
                        param.adamStep++;

                        for(int i = 0 ; i < param.outputFeatureDim ; i++){
                            double gradB = param.dB[i];
                            param.velocitydB[i] = beta1 * param.velocitydB[i] + (1.0 - beta1) * gradB;
                            param.adamSecondMomentdB[i] = beta2 * param.adamSecondMomentdB[i] + (1.0 - beta2) * gradB * gradB;

                            double mHatB = param.velocitydB[i] / (1.0 - std::pow(beta1, param.adamStep));
                            double vHatB = param.adamSecondMomentdB[i] / (1.0 - std::pow(beta2, param.adamStep));
                            param.bias[i] -= this->learningRate * (mHatB / (std::sqrt(vHatB) + epsilon));

                            for(int j = 0 ; j < param.inputFeatureDim ; j++){
                                double gradW = param.dW[i][j];
                                param.velocitydW[i][j] = beta1 * param.velocitydW[i][j] + (1.0 - beta1) * gradW;
                                param.adamSecondMomentdW[i][j] = beta2 * param.adamSecondMomentdW[i][j] + (1.0 - beta2) * gradW * gradW;

                                double mHatW = param.velocitydW[i][j] / (1.0 - std::pow(beta1, param.adamStep));
                                double vHatW = param.adamSecondMomentdW[i][j] / (1.0 - std::pow(beta2, param.adamStep));
                                param.weights[i][j] -= this->learningRate * (mHatW / (std::sqrt(vHatW) + epsilon));
                            }
                        }
                    }
                }
                else{
                    throw std::runtime_error{"Invalid Optimizer"};
                }
            }
            
            void saveModel(const std::string &fileName){

                //Create File
                std::ofstream file(fileName, std::ios::binary);
                if(!file){
                    throw std::runtime_error{"Couldnot open file for writing"};
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
                    file.write((char *)(&params.activationType), sizeof(params.activationType));
                    
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

            void loadModel(const std::string &fileName){
                // Open File
                std::ifstream file(fileName, std::ios::binary);
                if(!file){
                    throw std::runtime_error {"Could not open model file for reading"};
                }

                // Check Magic number
                char magic[8];
                file.read(magic, sizeof(magic));

                if(std::string(magic) != "Network"){
                    throw std::runtime_error{"Invalid model file"};
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

                    layers.push_back(std::move(layerX));
                }
                file.close();
            }
    };
};