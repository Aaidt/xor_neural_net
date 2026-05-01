#define _USE_MATH_DEFINES

#include<iostream>
#include<vector>
#include<random>
#include<cmath>

double box_muller(){
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double u1 = dist(gen);
    double u2 = dist(gen);
    double pi = M_PI;

    double z = std::sqrt(-2 * std::log(u1)) * std::cos(2 * pi * u2);

    return z;
}

std::vector<std::vector<double>> randn(int rows, int cols){
    std::vector<std::vector<double>> v(rows, std::vector<double>(cols));
    for(int i = 0; i < rows; i ++){
        v[i] = {};
        for(int j = 0; j < cols; j++){
            v[i][j] = box_muller() * 0.5;
        }
    }

    return v;
}

std::vector<std::vector<double>> zeroes(int cols){
    return std::vector<std::vector<double>> (1, std::vector<double>(cols, 0.0));
}

int main (){
    std::vector<std::vector<int>> A = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<std::vector<int>> B = {
        {0},
        {1},
        {1},
        {0}
    };
        
    int INPUT_SIZE = 2;
    int HIDDEN_SIZE = 4;
    int OUTPUT_SIZE = 1;

    std::vector<std::vector<double>> weights_input_hidden = randn(INPUT_SIZE, HIDDEN_SIZE);
    std::vector<std::vector<double>> bias_hidden = zeroes(HIDDEN_SIZE);


    std::vector<std::vector<double>> weights_hidden_output = randn(HIDDEN_SIZE, OUTPUT_SIZE);
    std::vector<std::vector<double>> bias_output = zeroes(OUTPUT_SIZE);

    


    return 0;
}