#define _USE_MATH_DEFINES

#include<iostream>
#include<type_traits>
#include<vector>
#include<random>
#include<cmath>
#include<string>

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
        for(int j = 0; j < cols; j++){
            v[i][j] = box_muller() * 0.5;
        }
    }

    return v;
}

std::vector<std::vector<double>> zeroes(int cols){
    return std::vector<std::vector<double>> (1, std::vector<double>(cols, 0.0));
}

template <typename T>
void print_matrix(const std::string& label, const std::vector<std::vector<T>>& matrix){
    std::cout << label << ": \n";
    for(const auto& rows : matrix){
        for(const auto& val : rows){
            std::cout << "| " << val << " | ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

template <typename T>
std::vector<std::vector<T>> sigmoid(const std::vector<std::vector<T>>& matrix){
    static_assert(std::is_floating_point<T>::value, "sigmoid requires a floating-point type");

    std::vector<std::vector<T>> result = matrix

    for(auto& rows : matrix){
        for(auto& vals : rows){
            vals = 1.0 / (1.0 + (std::exp(-vals)));
        }
    }

    return result;
}

template <typename T>
std::vector<std::vector<T>> sigmoid_derivative(const std::vector<std::vector<T>>& sigmoid_output){
    static_assert(std::is_floating_point<T>::value, "sigmoid derivative requires a floating-point type");

    std::vector<std::vector<T>> result = sigmoid_output;

    for(auto& rows : m){
        for(auto& val : rows){
            val = val * (1.0 - val);
        }
    }

    return result;
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
    print_matrix("weights_input_hidden", weights_input_hidden);
    print_matrix("bias_hidden", bias_hidden);
    


    std::vector<std::vector<double>> weights_hidden_output = randn(HIDDEN_SIZE, OUTPUT_SIZE);
    std::vector<std::vector<double>> bias_output = zeroes(OUTPUT_SIZE);
    print_matrix("weights_hidden_output", weights_hidden_output);
    print_matrix("bias_output", bias_output);
    


    return 0;
}