#include <iostream>
#include <vector>
#include <random>

using namespace std;

enum class ActivationType{
    ReLU,
    Sigmoid,
    Softmax,
    Tanh
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

class Layer{
    private:
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
            //! For Softmax is not Done
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
            
            vector<vector<double>> dl_dz(this->batchSize, vector<double>(this->outputFeatureDim));

            for(int i = 0 ; i < this->batchSize ; i++){
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

        vector<vector<double>> feedforward(vector<vector<double>> &input){
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
                vector<vector<double>> prediction = feedforward(input);

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
                vector<vector<double>> prediction = feedforward(input);

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


};

int main(){
    vector<vector<double>> input = {{100, 2, 3},{4, 50, -6}};
    vector<vector<double>> y = {{1.0}, {2.0}};

    Network nn(LossFunction :: MSELoss, Optimizer :: Adam, 0.001);

    nn.addLayer(input[0].size(), 2, ActivationType :: ReLU, 2);
    nn.addLayer(2, 1, ActivationType :: ReLU, 2);

    vector<vector<double>> output = nn.feedforward(input);
    
    for(int i = 0 ; i < output.size() ; i++){
        for(double val : output[i]){
            cout << val << "\t\t";
        }
        cout << endl;
    }

double loss = nn.loss(input, y);
cout << "Loss :" << loss << endl;

}