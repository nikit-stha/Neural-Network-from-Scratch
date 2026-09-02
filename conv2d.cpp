#include <iostream>
#include "network.hpp"
#include <vector>

using namespace nn;
using namespace std;

int main(){
    Conv2DLayer conv2D(3, 3, 2, 1, ActivationType :: Linear);
    std::vector<std::vector<std::vector<double>>> input = {{{1, 2, 3},
                                            {4, 5, 6},
                                            {7, 8, 9}},
                                            {{10, 11, 12},
                                            {13, 14, 15},
                                            {16, 17, 18}},
                                            {{19, 20, 21},
                                            {22, 23, 24},
                                            {25, 26, 27}}};
                
    vector<vector<vector<double>>> output = conv2D.forward(input);

    for(int i = 0 ; i < output.size() ; i++){
        for(int j = 0 ; j < output[0].size() ; j++){
            for(int k = 0 ; k < output[0][0].size() ; k++){
                cout << output[i][j][k] << " ";
            }
        }
    }
}