#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

vector<vector<double>> concatenate_heads(vector<vector<double>>&, size_t, size_t, size_t);
void add_bias(vector<vector<double>>&, vector<double>&);
vector<vector<double>> genRandomMatrix(size_t, size_t);
vector<vector<double>> split_heads(vector<vector<double>>&, size_t, size_t);
vector<vector<double>> matmul(vector<vector<double>>&, vector<vector<double>>&);
void softmax_rows(vector<vector<double>>&);
vector<vector<double>> softmax(vector<vector<double>>&);
vector<vector<double>> matrixTranspose(vector<vector<double>>&);
vector<vector<double>> read_data(string, size_t, size_t);
vector<vector<double>> get_positional_encoding(size_t, size_t);
vector<vector<double>> add_vectors(vector<vector<double>>&, vector<vector<double>>&);
vector<vector<double>> layer_norm(vector<vector<double>>&, vector<double>&, vector<double>&, float);

class MultiHeadAttention {
    public:
        size_t d_model, num_heads, d_key, d_value;
        vector<vector<double>> WQ, WK, WV, WO;

        MultiHeadAttention(size_t d_model, size_t num_heads) : d_model(d_model), num_heads(num_heads) {
            assert(d_model % num_heads == 0);
            this->d_model = d_model;
            this->num_heads = num_heads;
            this->d_key = d_model / num_heads;
            this->d_value = d_model / num_heads;
            this->WQ = genRandomMatrix(d_model, d_model);
            this->WK = genRandomMatrix(d_model, d_model);
            this->WV = genRandomMatrix(d_model, d_model);
            this->WO = genRandomMatrix(d_model, d_model);
        }

        vector<vector<double>> forward(vector<vector<double>>& x) {
            size_t seq_len = x.size();

            vector<vector<double>> Q = matmul(x, WQ);
            vector<vector<double>> K = matmul(x, WK);
            vector<vector<double>> V = matmul(x, WV);

            vector<vector<double>> Q_heads = split_heads(Q, num_heads, d_key);
            vector<vector<double>> K_heads = split_heads(K, num_heads, d_key);
            vector<vector<double>> V_heads = split_heads(V, num_heads, d_value);

            vector<vector<double>> attention_heads(num_heads * seq_len);

            #pragma omp parallel for 
            for (size_t i = 0; i < num_heads; i++) {
                vector<vector<double>> Q_head(seq_len), K_head(seq_len), V_head(seq_len);
                for (size_t j = 0; j < seq_len; j++) {
                    Q_head[j] = Q_heads[i * seq_len + j];
                    K_head[j] = K_heads[i * seq_len + j];
                    V_head[j] = V_heads[i * seq_len + j];
                }

                vector<vector<double>> K_head_T = matrixTranspose(K_head);
                vector<vector<double>> attention_scores = matmul(Q_head, K_head_T);

                double scale = sqrt(d_key);
                for (auto& row : attention_scores) {
                    for (auto& val : row) {
                        val /= scale;
                    }
                }

                softmax_rows(attention_scores);

                vector<vector<double>> attention_output = matmul(attention_scores, V_head);

                for (size_t j = 0; j < seq_len; j++) {
                    attention_heads[i * seq_len + j] = move(attention_output[j]);
                }
            }

            vector<vector<double>> concat = concatenate_heads(attention_heads, num_heads, seq_len, d_value);

            vector<vector<double>> output = matmul(concat, WO);

            return output;
        }
};

class FeedForward {
    public:
        size_t d_model, ff_dim;
        vector<vector<double>> W1, W2;
        vector<double> b1, b2;

        FeedForward(size_t d_model, size_t ff_dim) : d_model(d_model), ff_dim(ff_dim) {
            this->W1 = genRandomMatrix(d_model, ff_dim);
            this->b1 = vector<double>(ff_dim, 0.0f);
            this->W2 = genRandomMatrix(ff_dim, d_model);
            this->b2 = vector<double>(d_model, 0.0f);
        }

        vector<vector<double>> forward(vector<vector<double>>& x) {
            // Layer 1
            vector<vector<double>> h1 = matmul(x, W1);
            add_bias(h1, b1);

            // ReLU activation
            #pragma omp parallel for
            for (auto& row : h1) {
                for (auto& val : row) {
                    val = max(0.0, val);
                }
            }

            // Layer 2
            vector<vector<double>> h2 = matmul(h1, W2);
            add_bias(h2, b2);

            return h2;
        }
};

vector<vector<double>> concatenate_heads(vector<vector<double>>& x, size_t num_heads, size_t seq_len, size_t d_value) {
    vector<vector<double>> X(seq_len, vector<double>(num_heads * d_value, 0.0f));
    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (size_t i = 0; i < num_heads; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            for (size_t k = 0; k < d_value; k++) {
                X[j][i * d_value + k] = x[i * seq_len + j][k];
            }
        }
    }
    return X;
}

void add_bias(vector<vector<double>>& x, vector<double>& b) {
    assert(x[0].size() == b.size());
    #pragma omp parallel for
    for (auto& row : x) {
        for (size_t i = 0; i < row.size(); i++) {
            row[i] += b[i];
        }
    }
}

vector<vector<double>> genRandomMatrix(size_t rows, size_t cols) {
    vector<vector<double>> matrix(rows, vector<double>(cols, 0.0));
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
    return matrix;
}

vector<vector<double>> split_heads(vector<vector<double>>& x, size_t num_heads, size_t d_head) {
    size_t seq_len = x.size();
    size_t d_model = x[0].size();
    vector<vector<double>> X_split(seq_len * num_heads, vector<double>(d_head, 0.0f));

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < d_head; ++j) {
                X_split[h * seq_len + i][j] = x[i][h * d_head + j];
            }
        }
    }
    return X_split;
}

vector<vector<double>> matmul(vector<vector<double>>& a, vector<vector<double>>& b) {
    size_t n = a.size(), m = a[0].size(), p = b[0].size();
    vector<vector<double>> c(n, vector<double>(p, 0.0));
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < p; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < m; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    return c;
}

void softmax_rows(vector<vector<double>>& a) {
    #pragma omp parallel for
    for (auto& row : a) {
        double max_val = *max_element(row.begin(), row.end());
        double sum = 0.0f;
        for (auto& val : row) {
            val = exp(val - max_val);
            sum += val;
        }
        for (auto& val : row) {
            val /= sum;
        }
    }
}

vector<vector<double>> softmax(vector<vector<double>>& a){
    #pragma omp parallel for
    for(auto& row : a){
        double max_val = *max_element(row.begin(), row.end());
        double sum = 0.0f;
        for(auto& val : row){
            val = exp(val - max_val);
            sum += val;
        }
        for(auto& val : row){
            val /= sum;
        }
    }
    return a;
}

vector<vector<double>> matrixTranspose(vector<vector<double>>& a){
    vector<vector<double>> b(a[0].size(), vector<double>(a.size(), 0));
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < a[0].size(); j++) {
            b[j][i] = a[i][j];
        }
    }
    return b;
}

vector<vector<double>> read_data(string filename, size_t sequence_length, size_t embed_dim) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "File not found!" << endl;
        return {};
    }
    vector<vector<double>> data;
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> vec(embed_dim, 0);
        for (size_t i = 0; i < embed_dim; i++) {
            ss >> vec[i];
        }
        if(vec.size() != embed_dim){
            cout << "Mismatch in sample size. Skipping sample" << endl;
            continue;
        }
        data.push_back(move(vec)); // Use move to avoid copying
    }
    return data;
}

vector<vector<double>> get_positional_encoding(size_t sequence_length, size_t d_model) {
    vector<vector<double>> positional_encodings(sequence_length, vector<double>(d_model, 0.0f));
    #pragma omp parallel for
    for (size_t pos = 0; pos < sequence_length; pos++) {
        for (size_t i = 0; i < d_model; i++) {
            if (i % 2 == 0) {
                positional_encodings[pos][i] = sin(pos / pow(10000, (double)i / d_model));
            } else {
                positional_encodings[pos][i] = cos(pos / pow(10000, (double)(i - 1) / d_model));
            }
        }
    }
    return positional_encodings;
}

vector<vector<double>> add_vectors(vector<vector<double>>& a, vector<vector<double>>& b) {
    vector<vector<double>> c(a.size(), vector<double>(a[0].size(), 0.0f));
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < a[0].size(); j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    return c;
}

vector<vector<double>> layer_norm(vector<vector<double>>& input, vector<double>& gamma, vector<double>& beta, float epsilon = 1e-6) {
    size_t seq_len = input.size();
    size_t dim = input[0].size();
    vector<vector<double>> output(seq_len, vector<double>(dim, 0.0f));
    #pragma omp parallel for 
    for (size_t i = 0; i < seq_len; ++i) {
        double mean = 0.0f;
        for (auto val : input[i]) mean += val;
        mean /= dim;

        double var = 0.0f;
        for (double val : input[i]) var += (val - mean) * (val - mean);
        var /= dim;

        for (size_t j = 0; j < dim; ++j) {
            output[i][j] = gamma[j] * ((input[i][j] - mean) / sqrt(var + epsilon)) + beta[j];
        }
    }
    return output;
}

class EncoderLayer {
    public:
        size_t d_model;
        size_t num_heads;
        size_t ff_d;

        MultiHeadAttention mha;
        FeedForward ff;
        
        vector<double> beta;
        vector<double> gamma;
        
        EncoderLayer(size_t d_model, size_t num_heads, size_t ff_d) : d_model(d_model), num_heads(num_heads), ff_d(ff_d), mha(d_model, num_heads), ff(d_model, ff_d) {
            beta = vector<double>(d_model, 0.0f);
            gamma = vector<double>(d_model, 1.0f);
        }

        vector<vector<double>> forward(vector<vector<double>>& x) {
            vector<vector<double>> attn_output = mha.forward(x);

            vector<vector<double>> addLayer1 = add_vectors(x, attn_output);
            vector<vector<double>> norm1 = layer_norm(addLayer1, gamma, beta, 1e-6);

            vector<vector<double>> ff_output = ff.forward(norm1);

            vector<vector<double>> addLayer2 = add_vectors(norm1, ff_output);
            vector<vector<double>> norm2 = layer_norm(addLayer2, gamma, beta);

            return norm2;
        }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <num_threads>\n";
        return 1;
    }

    int num_threads = stoi(argv[1]);  // Get number of threads from command-line argument
    omp_set_num_threads(num_threads);

    // transformer parameters
    size_t d_model = 500;
    size_t embed_dim = 500;
    size_t sequence_length = 100;
    size_t num_heads = 50;
    size_t ff_dim = 1280;

    string input_file = "dataset_vectors.txt";

    cout << "Loading the data from the file..." << endl;

    vector<vector<double>> input_vectors = read_data(input_file, sequence_length, embed_dim);
    if(input_vectors.empty()){
        cout << "No data found in the file. Exiting..." << endl;
        return 1;
    }
    cout << "Data read successfully with sequence length "<< input_vectors.size() << endl;

    double start_time = omp_get_wtime();

    vector<double> sample = input_vectors[0];
    vector<vector<double>> sample_vector(sequence_length, vector<double>(embed_dim, 0.0f));

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(size_t i = 0; i< sequence_length; i++){
        for(size_t j = 0; j < embed_dim; j++){
            sample_vector[i][j] = sample[i*embed_dim + j];
        }
    }

    vector<vector<double>> positional_encodings = get_positional_encoding(sequence_length, d_model);
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(size_t i = 0; i < sequence_length; i++){
        for(size_t j = 0; j < d_model; j++){
            sample_vector[i][j] += positional_encodings[i][j];
        }
    }

    EncoderLayer encoder_layer(d_model, num_heads, ff_dim);

    vector<vector<double>> encoder_output = encoder_layer.forward(sample_vector);

    cout << "Output after input data is passed through the transformer : " << endl;
    ofstream outputFile("output.txt");

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    cout << "Threads: " << num_threads << " | Time: " << elapsed_time << " seconds.\n";

    ofstream output_file("execution_times.txt", ios::app);
    output_file << num_threads << " " << elapsed_time << "\n";
    output_file.close();
    
    for (size_t i = 0; i < encoder_output.size(); ++i) {
        for (size_t j = 0; j < encoder_output[i].size(); ++j) {
            {
                outputFile << encoder_output[i][j] << " ";
                cout << encoder_output[i][j] << " ";
            }
        }
        {
            cout << endl;
        }
    }

    return 0;
}