#include <iostream>
#include <vector>
#include <random>

using namespace std;

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

struct Parameters{
    vector<vector<double>> weights;
    vector<double> bias;

    vector<vector<double>> dW;
    vector<double> dB;
};

class Layer{
    public:         //! Change to Private
        int inputFeatureDim;
        int outputFeatureDim;
        ActivationType activationFunction;

        vector<vector<double>> weights;
        vector<double> bias;

        int batchSize;

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
        Layer(int inputFeatureDim, int outputFeatureDim, ActivationType activationFunction, int batchSize){
            this->inputFeatureDim = inputFeatureDim;
            this->outputFeatureDim = outputFeatureDim;
            this->activationFunction = activationFunction;

            this->batchSize = batchSize;

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
            else if(this->activationFunction == ActivationType :: Softmax){
                double sum = 0;
                for(int i = 0 ; i < z.size() ; i++){
                    sum += exp(z[i]);
                }
                for(int i = 0 ; i < z.size() ; i++){
                    output[i] = (double) (exp(z[i]) / sum);
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
            else if(this->activationFunction == ActivationType :: Softmax){
                //! Considering Softmax to be used with CrossCategoricalEntropyLoss
                vector<double> softmax = activation(z);
                for(int i = 0 ; i < z.size() ; i++){
                    derivative[i] = softmax[i] - z[i];
                }
            }
            else{
                cout << "Incorrect Activation Function" << endl;
                return {};
            }
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

            for(int i = 0 ; i < batchSize ; i++){
                vector<double> rows = activationDerivative(this->lastZ[i]);
                for(int j = 0 ; j < this->outputFeatureDim ; j++){
                    dl_dz[i][j] = dl_da[i][j] * rows[j];
                }
            }

            for (int j = 0; j < outputFeatureDim; j++) {
                for (int k = 0; k < inputFeatureDim; k++) {
                    double sum = 0.0;
                    for (int i = 0; i < batchSize; i++) {
                        sum += dl_dz[i][j] * lastInput[i][k];
                    }
                    this->dW[j][k] = sum / batchSize;
                }
            }

            for (int j = 0; j < outputFeatureDim; j++) {
                double sum = 0.0;
                for (int i = 0; i < batchSize; i++) {
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
        
        void addLayer(int inputFeatureDim, int outputFeatureDim, ActivationType activationType, int batchSize){
            layers.emplace_back(inputFeatureDim, outputFeatureDim, activationType, batchSize);
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
                double sum = 0;
                vector<vector<double>> prediction = forwardpass(input);

                for(int i = 0 ; i < batchSize ; i++){
                    for(int j = 0 ; j < prediction[0].size() ; j++){
                        sum += (double) ((1.0 / 2.0) * (prediction[i][j] - output[i][j]) * (prediction[i][j] - output[i][j]));
                    }
                }
                return (double)(sum / batchSize);
            }
            else if(this->lossFunction == LossFunction :: CrossEntropyLoss){
                //! Cross Entropy Loss Not Completed
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
            }
        }

        void backwardPass(vector<vector<double>> &input, vector<vector<double>> &target) {
            vector<vector<double>> y_pred = forwardpass(input);
            int bSize = input.size();
            int outDim = target[0].size();

            vector<vector<double>> dl_da(bSize, vector<double>(outDim, 0.0));

            if (this->lossFunction == LossFunction::MSELoss) {
                for (int i = 0; i < bSize; i++) {
                    for (int j = 0; j < outDim; j++) {
                        dl_da[i][j] = y_pred[i][j] - target[i][j];
                    }
                }
            } 
            else if (this->lossFunction == LossFunction::MAELoss) {
                for (int i = 0; i < bSize; i++) {
                    for (int j = 0; j < outDim; j++) {
                        if (y_pred[i][j] > target[i][j]) dl_da[i][j] = 1.0;
                        else if (y_pred[i][j] < target[i][j]) dl_da[i][j] = -1.0;
                        else dl_da[i][j] = 0.0;
                    }
                }
            }

            //! Other Loss Function is not Complete

            vector<vector<double>> currentGradient = dl_da;
            if (this->optimizer == Optimizer::SGD) {
                for (int l = (int)layers.size() - 1; l >= 0; l--) {
                    currentGradient = layers[l].backward(currentGradient);
                    for (int i = 0; i < layers[l].outputFeatureDim; i++) {
                        layers[l].bias[i] -= this->learningRate * layers[l].dB[i];

                        for (int j = 0; j < layers[l].inputFeatureDim; j++) {
                            layers[l].weights[i][j] -= this->learningRate * layers[l].dW[i][j];
                        }
                    }
                }
            }
        }
        
};

int batchSize = 2;

int main(){
    vector<vector<double>> data = {{1.0, 2.0, 3.0}, 
                                    {4.0, 5.0, 6.0},
                                     {7.0, 8.0, 9.0},
                                      {10.0, 11.0, 12.0},
                                       {13.0, 14.0, 15.0},
                                        {16.0, 17.0, 18.0}};

    vector<vector<double>> output = {{1.0},
                                    {2.0},
                                    {3.0},
                                    {4.0},
                                    {5.0},
                                    {6.0}};

    int inputFeatureSize = data[0].size(), outputFeatureSize = 1;

    Network nn(LossFunction :: MSELoss, Optimizer :: SGD, 0.001);
    nn.addLayer(inputFeatureSize, 5, ActivationType :: ReLU, batchSize);
    nn.addLayer(5, outputFeatureSize, ActivationType :: Linear, batchSize);


    vector<vector<double>> testX = {{17.0, 18.0, 19.0}};
    int epochs = 1000;

    for(int i = 0 ; i < epochs ; i++){
        nn.backwardPass(data, output);
    }

    vector<vector<double>> prediction = nn.forwardpass(testX);
    for(int i = 0 ; i < prediction.size() ; i++){
        for(double val : prediction[i]){
            cout << val << " ";
        }
        cout << endl;
    }
}