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

template <typename T>
T compute_loss(
    const std::vector<std::vector<T>>& y_true, 
    const std::vector<std::vector<T>>& y_pred 
){
    static_assert(std::is_floating_point<T>::value, "compute_loss requires floating-point type");

    if(y_true.empty() || y_pred.empty() || y_true[0].size() != y_pred[0].size() || y_true.size() != y_pred.size()){
        throw std::invalid_argument("Matrices for computing loss cannot be empty");
    }

    int n = y_true.size();
    int m = y_true[0].size();

    T sum = 0.0;

    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            T diff = (y_true[i][j] - y_pred[i][j]);
            sum += diff * diff;
        }
    }

    return sum / (n * m);
}

template <typename T>
std::vector<std::vector<T>> hadamard_product(
    const std::vector<std::vector<T>>& A,
    const std::vector<std::vector<T>>& B
){
    static_assert(std::is_floating_point<T>::value, "hadamard_product requires floating-point type");
    if(A.empty() || A[0].empty() || B.empty() || B[0].empty()){
        throw std::invalid_argument("Matrices cant be empty");
    }

    int n = A.size();
    int m = A[0].size();

    if(B.size() != n || B[0].size() != m){
        throw std::invalid_argument("Matrices must have the same dimensions");
    }

    for(const auto& rows : A){
        if(rows.size() != m){
            throw std::invalid_argument("A is not rectangular");
        }
    }

    for(const auto& rows : B){
        if(rows.size() != m){
            throw std::invalid_argument("B is not rectangular");
        }
    }

    std::vector<std::vector<T>> result(n, std::vector<T>(m));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            result[i][j] = A[i][j] * B[i][j];
        }
    }

    return result;
}

template <typename T>
std::vector<std::vector<T>> elementwise_sub(
    const std::vector<std::vector<T>>& A,
    const std::vector<std::vector<T>>& B
){
    static_assert(std::is_floating_point<T>::value, "hadamard_product requires floating-point type");

    int n = A.size();
    int m = A[0].size();
    if(A.size() != B.size() || A[0].size() != B[0].size()){
        throw std::invalid_argument("Both matrices should have the same dimensions");
    }

    for(const auto& rows : A){
        if(rows.size() != m){
            throw std::invalid_argument("A is not rectangular");
        }
    }

    for(const auto& rows : B){
        if(rows.size() != m){
            throw std::invalid_argument("B is not rectangular");
        }
    }

    std::vector<std::vector<T>> result(n, std::vector<T>(m));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            result[i][j] = A[i][j] - B[i][j];
        }
    }

    return result;
}

template <typename T>
std::vector<std::vector<T>> transpose(
    const std::vector<std::vector<T>>& matrix
){
    if(matrix.empty() || matrix[0].empty()){
        throw std::invalid_argument("Matrix cannot be empty");
    }

    int n = matrix.size();
    int m = matrix[0].size();
    for(const auto& rows : matrix){
        if(rows.size() != m){
            throw std::invalid_argument("Matrix is not reactangular");
        }
    }

    std::vector<std::vector<T>> result(m, std::vector<T>(n));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}

template <typename T>
std::vector<std::vector<T>> scalar_operation(
    const std::vector<std::vector<T>>& matrix,
    T val,
    std::string op
){
    if(matrix.empty() || matrix[0].empty()){
        throw std::invalid_argument("Matrix cannot be empty");
    }

    if(op != "multiply" && op != "divide"){
        throw std::invalid_argument("operation must be either multiply or divide");
    }

    int n = matrix.size();
    int m = matrix[0].size();
    for(const auto& rows : matrix){
        if(rows.size() != m){
            throw std::invalid_argument("Matrix is not reactangular");
        }
    }

    std::vector<std::vector<T>> result = matrix;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            if(op == "divide"){
                result[i][j] = matrix[i][j] / val;
            }else if(op == "multiply"){
                result[i][j] = matrix[i][j] * val;
            }
        }
    }

    return result;
}

template <typename T>
std::vector<std::vector<T>> sum_rows(
    const std::vector<std::vector<T>>& matrix
){
    if(matrix.empty() || matrix[0].empty()){
        throw std::invalid_argument("Matrix cannot be empty");
    }

    int n = matrix.size();
    int m = matrix[0].size();
    for(const auto& rows : matrix){
        if(rows.size() != m){
            throw std::invalid_argument("Matrix is not reactangular");
        }
    }

    std::vector<std::vector<T>> result(1, std::vector<T>(m, 0));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            result[0][j] += matrix[i][j];
        }
    }

    return result;
}


template <typename T>
void backward_pass(
    const std::vector<std::vector<T>>& A,
    const std::vector<std::vector<T>>& B,
    const std::vector<std::vector<T>>& z_hidden,
    const std::vector<std::vector<T>>& a_hidden,
    const std::vector<std::vector<T>>& z_output,
    const std::vector<std::vector<T>>& a_output,
    std::vector<std::vector<T>>& weights_hidden_output,
    std::vector<std::vector<T>>& bias_output,
    std::vector<std::vector<T>>& weights_input_hidden,
    std::vector<std::vector<T>>& bias_hidden,
    double learning_rate
){
    static_assert(std::is_floating_point<T>::value, "backward_pass requires floating-point type");

    T val = static_cast<T>(A.size());

    // OUTPUT LAYER
    std::vector<std::vector<T>> output_error = elementwise_sub(a_output, B);
    std::vector<std::vector<T>> output_delta = hadamard_product(output_error, sigmoid_derivative(a_output));

    std::vector<std::vector<T>> grad_weights_hidden_output = scalar_operation(
        mat_mul(transpose(a_hidden), output_delta),
        val,
        "divide"
    );

    std::vector<std::vector<T>> grad_bias_output = scalar_operation(
        sum_rows(output_delta),
        val,
        "divide"
    );

    // HIDDEN LAYER
    std::vector<std::vector<T>> hidden_error = mat_mul(
        output_delta,
        transpose(weights_hidden_output)
    );
    std::vector<std::vector<T>> hidden_delta = hadamard_product(
        hidden_error,
        sigmoid_derivative(a_hidden)
    );

    std::vector<std::vector<T>> grad_weights_input_hidden = scalar_operation(
        mat_mul(transpose(A), hidden_delta),
        val,
        "divide"
    );
    std::vector<std::vector<T>> grad_bias_hidden = scalar_operation(
        sum_rows(hidden_delta),
        val,
        "divide"
    );

    // UPDATE WEIGHTS
    weights_hidden_output = elementwise_sub(
        weights_hidden_output,
        scalar_operation(grad_weights_hidden_output, learning_rate, "multiply")
    );
    bias_output = elementwise_sub(
        bias_output,
        scalar_operation(grad_bias_output, learning_rate, "multiply")
    );
    weights_input_hidden = elementwise_sub(
        weights_input_hidden,
        scalar_operation(grad_weights_input_hidden, learning_rate, "multiply")
    );
    bias_hidden = elementwise_sub(
        bias_hidden,
        scalar_operation(grad_bias_hidden, learning_rate, "multiply")
    );

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
    std::cout << "Predictions are garbage — the network hasn't learned anything yet." << std::endl;

    double loss = compute_loss(B, fw_pass.a_output);
    std::cout << "Loss on the first pass: " << loss << std::endl;

    double LEARNING_RATE = 2.0;
    int iterations = 10000;
    std::vector<double> loss_history;

    std::cout << "Starting training... \n";
    std::cout << "---------------------------------- \n";

    for(int i = 0; i < iterations; i++){
        auto fw = forward_pass(A, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output);
        
        double loss = compute_loss(fw.a_output, B);
        
        loss_history.push_back(loss);

        backward_pass(
            A, 
            B, 
            fw.z_hidden, 
            fw.a_hidden, 
            fw.z_output, 
            fw.a_output, 
            weights_hidden_output,
            bias_output,
            weights_input_hidden,
            bias_hidden,
            LEARNING_RATE
        );

        if(i % 1000 == 0){
            std::cout << "Iteration: " << i << " | loss: " << loss << "\n";
        }
    }

    auto fw = forward_pass(A, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output);
    std::cout << "Final results after training: \n";
    std::cout << "-------------------------------------- \n";
    
    for(const auto& rows : fw.a_output){
        for(const auto& vals : rows){
            std::cout << std::round(vals) << " ";
        }
        std::cout << "\n";
    }


    return 0;
}