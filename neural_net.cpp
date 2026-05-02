#define _USE_MATH_DEFINES

#include<iostream>
#include<type_traits>
#include<stdexcept>
#include<vector>
#include<random>
#include<cmath>
#include<string>

double box_muller(){
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_real_distribution<double> dist(1e-9, 1.0);

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
void print_matrix(
    const std::string& label, 
    const std::vector<std::vector<T>>& matrix
){
    std::cout << label << ": \n";
    for(const auto& rows : matrix){
        for(const auto& val : rows){
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

template <typename T>
std::vector<std::vector<T>> sigmoid(
    const std::vector<std::vector<T>>& matrix
){
    static_assert(std::is_floating_point<T>::value, "sigmoid requires a floating-point type");

    std::vector<std::vector<T>> result = matrix;

    for(auto& rows : result){
        for(auto& vals : rows){
            vals = 1.0 / (1.0 + (std::exp(-vals)));
        }
    }
    return result;
}

template <typename T>
std::vector<std::vector<T>> sigmoid_derivative(
    const std::vector<std::vector<T>>& sigmoid_output
){
    static_assert(std::is_floating_point<T>::value, "sigmoid derivative requires a floating-point type");

    std::vector<std::vector<T>> result = sigmoid_output;

    for(auto& rows : result){
        for(auto& val : rows){
            val = val * (1.0 - val);
        }
    }
    return result;
}

template <typename T>
std::vector<std::vector<T>> mat_mul(
    const std::vector<std::vector<T>>& a,
    const std::vector<std::vector<T>>& b
){
    static_assert(std::is_floating_point<T>::value, "matrix multiplication requires floating-point type");

    if(a.empty() || a[0].empty() || b.empty() || b[0].empty()){
        throw std::invalid_argument("empty matrix");
    }

    int rows_a = a.size();
    int cols_a = a[0].size();
    int rows_b = b.size();
    int cols_b = b[0].size();
    
    if(cols_a != rows_b){
        throw std::invalid_argument("cols_a must equal rows_b");
    }

    std::vector<std::vector<T>> result(rows_a, std::vector<T>(cols_b, 0));

    for(int i = 0; i < rows_a; i++){
        for(int j = 0; j < cols_b; j++){
            for(int k = 0; k < cols_a; k++){
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

template <typename T>
std::vector<std::vector<T>> add_bias(
    const std::vector<std::vector<T>>& matrix, 
    const std::vector<std::vector<T>>& bias
){
    static_assert(std::is_floating_point<T>::value, "adding bias requires floating-point type");

    if(matrix.empty() || matrix[0].empty()){
        throw std::invalid_argument("matrix cannot be empty");
    }

    int n = matrix.size();
    int m = matrix[0].size();

    for(const auto& rows : matrix){
        if(rows.size() != m){
            throw std::invalid_argument("Matrix rows must have equal size");
        }
    }

    if(bias.size() != 1 || bias[0].size() != m){
        throw std::invalid_argument("bias must be 1 x m matrix");
    }

    std::vector<std::vector<T>> result = matrix;

    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            result[i][j] += bias[0][j];
        }
    }

    return result;
}

template <typename T>
struct Forward {
    std::vector<std::vector<T>> z_hidden;
    std::vector<std::vector<T>> a_hidden;
    std::vector<std::vector<T>> z_output;
    std::vector<std::vector<T>> a_output;
};

template <typename T>
Forward<T> forward_pass(
    const std::vector<std::vector<T>>& A,
    const std::vector<std::vector<T>>& w1,
    const std::vector<std::vector<T>>& b1,
    const std::vector<std::vector<T>>& w2,
    const std::vector<std::vector<T>>& b2
){
    static_assert(std::is_floating_point<T>::value, "forward_pass requires floating-point type");
    Forward<T> out;

    // input => hidden
    out.z_hidden = add_bias(mat_mul(A, w1), b1);
    out.a_hidden = sigmoid(out.z_hidden);

    // hidden => output
    out.z_output = add_bias(mat_mul(out.a_hidden, w2), b2);
    out.a_output = sigmoid(out.z_output);

    return out;
} 

int main (){
    std::vector<std::vector<double>> A = {
        {0.0, 1.0},
        {0.0, 0.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    std::vector<std::vector<double>> B = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
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
    
    auto fw_pass = forward_pass(A, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output);
    std::cout << "Forward pass on an untrained network: \n" << "---------------------------------------------- \n";
    for(const auto& rows : fw_pass.a_output){
        for(const auto& vals : rows){
            std::cout << vals << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    return 0;
}