#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>

using namespace std;

vector<vector<double>> concatenate_heads(vector<vector<double>>&, size_t, size_t, size_t, int, int);
void add_bias(vector<vector<double>>&, vector<double>&, int, int);
vector<vector<double>> genRandomMatrix(size_t, size_t);
vector<vector<double>> split_heads(vector<vector<double>>&, size_t, size_t, int, int);
vector<vector<double>> matmul(vector<vector<double>>&, vector<vector<double>>&, int, int);
void softmax_rows(vector<vector<double>>&, int, int);
vector<vector<double>> softmax(vector<vector<double>>&, int, int);
vector<vector<double>> matrixTranspose(vector<vector<double>>&, int, int);
vector<vector<double>> read_data(string, size_t, size_t, int, int);
vector<vector<double>> get_positional_encoding(size_t, size_t, int, int);
vector<vector<double>> add_vectors(vector<vector<double>>&, vector<vector<double>>&, int, int);
vector<vector<double>> layer_norm(vector<vector<double>>& input, vector<double>& gamma, vector<double>& beta, float epsilon, int rank, int size);

vector<vector<double>> genRandomMatrix(size_t rows, size_t cols) {
    vector<vector<double>> matrix(rows, vector<double>(cols, 0.0));
    mt19937 gen(42); // Fixed seed for reproducibility
    uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

void syncMatrix(vector<vector<double>>& matrix, int rank, int size) {
    const int MAX_CHUNK_SIZE = 1000000; // Maximum chunk size (adjust as needed)
    int rows = matrix.size();
    if (rows == 0) return;
    
    int cols = matrix[0].size();
    
    // Process the matrix in chunks of rows
    int chunk_rows = MAX_CHUNK_SIZE / cols;
    if (chunk_rows == 0) chunk_rows = 1; // Ensure at least one row per chunk
    
    for (int chunk_start = 0; chunk_start < rows; chunk_start += chunk_rows) {
        int chunk_end = min(chunk_start + chunk_rows, rows);
        int chunk_size = (chunk_end - chunk_start) * cols;
        
        // Flatten the chunk
        vector<double> send_buf(chunk_size, 0.0);
        vector<double> recv_buf(chunk_size, 0.0);
        
        for (int i = chunk_start; i < chunk_end; i++) {
            for (int j = 0; j < cols; j++) {
                send_buf[(i - chunk_start) * cols + j] = matrix[i][j];
            }
        }
        
        // Synchronize the chunk
        MPI_Allreduce(send_buf.data(), recv_buf.data(), chunk_size, 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Update the matrix from the result
        for (int i = chunk_start; i < chunk_end; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = recv_buf[(i - chunk_start) * cols + j];
            }
        }
    }
}

class MultiHeadAttention {
    public:
        size_t d_model, num_heads, d_key, d_value;
        vector<vector<double>> WQ, WK, WV, WO;
    
        MultiHeadAttention(size_t d_model, size_t num_heads) : d_model(d_model), num_heads(num_heads) {
            assert(d_model % num_heads == 0);
            this->d_key = d_model / num_heads;
            this->d_value = d_model / num_heads;
            this->WQ = genRandomMatrix(d_model, d_model);
            this->WK = genRandomMatrix(d_model, d_model);
            this->WV = genRandomMatrix(d_model, d_model);
            this->WO = genRandomMatrix(d_model, d_model);
        }
    
        vector<vector<double>> forward(vector<vector<double>>& x, int rank, int size) {
            size_t seq_len = x.size();
            
            // Matrix multiplications
            vector<vector<double>> Q(seq_len, vector<double>(d_model, 0.0));
            vector<vector<double>> K(seq_len, vector<double>(d_model, 0.0));
            vector<vector<double>> V(seq_len, vector<double>(d_model, 0.0));
            
            // Calculate work distribution
            size_t rows_per_process = (seq_len + size - 1) / size; // Ceiling division
            size_t start_row = rank * rows_per_process;
            size_t end_row = min(start_row + rows_per_process, seq_len);
            
            // Compute Q, K, V matrices locally
            for (size_t i = start_row; i < end_row; i++) {
                for (size_t j = 0; j < d_model; j++) {
                    double q_sum = 0.0, k_sum = 0.0, v_sum = 0.0;
                    for (size_t k = 0; k < d_model; k++) {
                        q_sum += x[i][k] * WQ[k][j];
                        k_sum += x[i][k] * WK[k][j];
                        v_sum += x[i][k] * WV[k][j];
                    }
                    Q[i][j] = q_sum;
                    K[i][j] = k_sum;
                    V[i][j] = v_sum;
                }
            }
            
            // Synchronize Q, K, V
            syncMatrix(Q, rank, size);
            syncMatrix(K, rank, size);
            syncMatrix(V, rank, size);
            
            // Split heads (we'll handle each head separately)
            vector<vector<vector<double>>> Q_split(num_heads, vector<vector<double>>(seq_len, vector<double>(d_key, 0.0)));
            vector<vector<vector<double>>> K_split(num_heads, vector<vector<double>>(seq_len, vector<double>(d_key, 0.0)));
            vector<vector<vector<double>>> V_split(num_heads, vector<vector<double>>(seq_len, vector<double>(d_value, 0.0)));
            
            // Split heads locally
            for (size_t h = 0; h < num_heads; h++) {
                for (size_t i = 0; i < seq_len; i++) {
                    for (size_t j = 0; j < d_key; j++) {
                        Q_split[h][i][j] = Q[i][h * d_key + j];
                        K_split[h][i][j] = K[i][h * d_key + j];
                        V_split[h][i][j] = V[i][h * d_key + j];
                    }
                }
            }
            
            // Process heads in parallel
            vector<vector<vector<double>>> attention_outputs(num_heads, vector<vector<double>>(seq_len, vector<double>(d_value, 0.0)));
            
            // Calculate which heads this process will handle
            size_t heads_per_process = (num_heads + size - 1) / size; // Ceiling division
            size_t start_head = rank * heads_per_process;
            size_t end_head = min(start_head + heads_per_process, num_heads);
            
            for (size_t h = start_head; h < end_head; h++) {
                // Compute attention scores
                vector<vector<double>> scores(seq_len, vector<double>(seq_len, 0.0));
                
                // Matrix multiplication for scores
                for (size_t i = 0; i < seq_len; i++) {
                    for (size_t j = 0; j < seq_len; j++) {
                        double dot_product = 0.0;
                        for (size_t k = 0; k < d_key; k++) {
                            dot_product += Q_split[h][i][k] * K_split[h][j][k];
                        }
                        scores[i][j] = dot_product / sqrt(d_key);
                    }
                }
                
                // Apply softmax to each row
                for (size_t i = 0; i < seq_len; i++) {
                    double max_val = -numeric_limits<double>::infinity();
                    for (size_t j = 0; j < seq_len; j++) {
                        max_val = max(max_val, scores[i][j]);
                    }
                    
                    double sum = 0.0;
                    for (size_t j = 0; j < seq_len; j++) {
                        scores[i][j] = exp(scores[i][j] - max_val);
                        sum += scores[i][j];
                    }
                    
                    for (size_t j = 0; j < seq_len; j++) {
                        scores[i][j] /= sum;
                    }
                }
                
                // Calculate attention output
                for (size_t i = 0; i < seq_len; i++) {
                    for (size_t j = 0; j < d_value; j++) {
                        double weighted_sum = 0.0;
                        for (size_t k = 0; k < seq_len; k++) {
                            weighted_sum += scores[i][k] * V_split[h][k][j];
                        }
                        attention_outputs[h][i][j] = weighted_sum;
                    }
                }
            }
            
            // Synchronize attention outputs across processes
            for (size_t h = 0; h < num_heads; h++) {
                for (size_t i = 0; i < seq_len; i++) {
                    vector<double> local_vec(d_value, 0.0);
                    vector<double> global_vec(d_value, 0.0);
                    
                    if (h >= start_head && h < end_head) {
                        local_vec = attention_outputs[h][i];
                    }
                    
                    MPI_Allreduce(local_vec.data(), global_vec.data(), d_value, 
                                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    
                    attention_outputs[h][i] = global_vec;
                }
            }
            
            // Concatenate heads
            vector<vector<double>> concat_output(seq_len, vector<double>(d_model, 0.0));
            for (size_t i = 0; i < seq_len; i++) {
                for (size_t h = 0; h < num_heads; h++) {
                    for (size_t j = 0; j < d_value; j++) {
                        concat_output[i][h * d_value + j] = attention_outputs[h][i][j];
                    }
                }
            }
            
            // Final projection
            vector<vector<double>> output(seq_len, vector<double>(d_model, 0.0));
            
            // Calculate which rows this process will handle
            for (size_t i = start_row; i < end_row; i++) {
                for (size_t j = 0; j < d_model; j++) {
                    double sum = 0.0;
                    for (size_t k = 0; k < d_model; k++) {
                        sum += concat_output[i][k] * WO[k][j];
                    }
                    output[i][j] = sum;
                }
            }
            
            // Synchronize output
            syncMatrix(output, rank, size);
            
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
        
            vector<vector<double>> forward(vector<vector<double>>& x, int rank, int size) {
                size_t seq_len = x.size();
                
                // Calculate work distribution
                size_t rows_per_process = (seq_len + size - 1) / size; // Ceiling division
                size_t start_row = rank * rows_per_process;
                size_t end_row = min(start_row + rows_per_process, seq_len);
                
                // First layer
                vector<vector<double>> h1(seq_len, vector<double>(ff_dim, 0.0));
                
                // Compute first layer locally
                for (size_t i = start_row; i < end_row; i++) {
                    for (size_t j = 0; j < ff_dim; j++) {
                        double sum = 0.0;
                        for (size_t k = 0; k < d_model; k++) {
                            sum += x[i][k] * W1[k][j];
                        }
                        h1[i][j] = sum + b1[j];
                        
                        // ReLU activation
                        if (h1[i][j] < 0.0) h1[i][j] = 0.0;
                    }
                }
                
                // Synchronize h1
                syncMatrix(h1, rank, size);
                
                // Second layer
                vector<vector<double>> h2(seq_len, vector<double>(d_model, 0.0));
                
                // Compute second layer locally
                for (size_t i = start_row; i < end_row; i++) {
                    for (size_t j = 0; j < d_model; j++) {
                        double sum = 0.0;
                        for (size_t k = 0; k < ff_dim; k++) {
                            sum += h1[i][k] * W2[k][j];
                        }
                        h2[i][j] = sum + b2[j];
                    }
                }
                
                // Synchronize h2
                syncMatrix(h2, rank, size);
                
                return h2;
            }
        };

vector<vector<double>> concatenate_heads(vector<vector<double>>& x, size_t num_heads, size_t seq_len, size_t d_value, int rank, int size) {
    vector<vector<double>> X(seq_len, vector<double>(num_heads * d_value, 0.0f));
    
    // Calculate work distribution
    size_t rows_per_process = (seq_len + size - 1) / size; // Ceiling division
    size_t start_row = rank * rows_per_process;
    size_t end_row = min(start_row + rows_per_process, seq_len);

    // Each process fills its portion of the result
    for (size_t j = start_row; j < end_row; j++) {
        for (size_t i = 0; i < num_heads; i++) {
            size_t idx = i * seq_len + j;
            if (idx < x.size() && x[idx].size() == d_value) {
                for (size_t k = 0; k < d_value; k++) {
                    X[j][i * d_value + k] = x[idx][k];
                }
            }
        }
    }

    // Synchronize all rows of X
    for (size_t i = 0; i < seq_len; i++) {
        vector<double> local_row = X[i];
        vector<double> global_row(num_heads * d_value, 0.0);
        
        MPI_Allreduce(local_row.data(), global_row.data(), num_heads * d_value, 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        X[i] = global_row;
    }

    return X;
}

void add_bias(vector<vector<double>>& x, vector<double>& b, int rank, int size) {
    assert(x[0].size() == b.size());
    
    // Calculate work distribution
    size_t rows_per_process = (x.size() + size - 1) / size; // Ceiling division
    size_t start_row = rank * rows_per_process;
    size_t end_row = min(start_row + rows_per_process, x.size());

    for (size_t i = start_row; i < end_row; i++) {
        for (size_t j = 0; j < x[i].size(); j++) {
            x[i][j] += b[j];
        }
    }

    // Synchronize x after adding bias
    for (size_t i = 0; i < x.size(); i++) {
        vector<double> local_row = x[i];
        vector<double> global_row(local_row.size(), 0.0);
        
        MPI_Allreduce(local_row.data(), global_row.data(), local_row.size(), 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Since all processes added bias to their portion, divide by size
        // to get the correct value (except for rows that only one process handled)
        if (i >= start_row && i < end_row) {
            for (size_t j = 0; j < global_row.size(); j++) {
                global_row[j] /= size;
            }
        }
        
        x[i] = global_row;
    }
}

vector<vector<double>> split_heads(vector<vector<double>>& x, size_t num_heads, size_t d_head, int rank, int size) {
    size_t seq_len = x.size();
    vector<vector<double>> X_split(seq_len * num_heads, vector<double>(d_head, 0.0f));
    
    // Calculate work distribution
    size_t heads_per_process = (num_heads + size - 1) / size; // Ceiling division
    size_t start_head = rank * heads_per_process;
    size_t end_head = min(start_head + heads_per_process, num_heads);

    for (size_t h = start_head; h < end_head; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < d_head; ++j) {
                X_split[h * seq_len + i][j] = x[i][h * d_head + j];
            }
        }
    }

    // Synchronize X_split across all processes
    for (size_t i = 0; i < seq_len * num_heads; i++) {
        vector<double> local_row = X_split[i];
        vector<double> global_row(d_head, 0.0);
        
        MPI_Allreduce(local_row.data(), global_row.data(), d_head, 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        X_split[i] = global_row;
    }

    return X_split;
}

vector<vector<double>> matmul(vector<vector<double>>& a, vector<vector<double>>& b, int rank, int size) {
    size_t n = a.size(), m = a[0].size(), p = b[0].size();
    vector<vector<double>> c(n, vector<double>(p, 0.0));
    
    // Calculate work distribution
    size_t rows_per_process = (n + size - 1) / size; // Ceiling division
    size_t start_row = rank * rows_per_process;
    size_t end_row = min(start_row + rows_per_process, n);

    for (size_t i = start_row; i < end_row; i++) {
        for (size_t j = 0; j < p; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < m; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }

    // Synchronize result matrix across all processes
    for (size_t i = 0; i < n; i++) {
        vector<double> local_row = c[i];
        vector<double> global_row(p, 0.0);
        
        MPI_Allreduce(local_row.data(), global_row.data(), p, 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        c[i] = global_row;
    }

    return c;
}

void softmax_rows(vector<vector<double>>& a, int rank, int size) {
    // Calculate work distribution
    size_t rows_per_process = (a.size() + size - 1) / size; // Ceiling division
    size_t start_row = rank * rows_per_process;
    size_t end_row = min(start_row + rows_per_process, a.size());

    for (size_t i = start_row; i < end_row; i++) {
        double max_val = -std::numeric_limits<double>::infinity();
        for (const auto& val : a[i]) {
            max_val = max(max_val, val);
        }
        
        // Share max value across processes
        double global_max;
        MPI_Allreduce(&max_val, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        
        double sum = 0.0f;
        for (auto& val : a[i]) {
            val = exp(val - global_max);
            sum += val;
        }
        
        // Share sum across processes
        double global_sum;
        MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        for (auto& val : a[i]) {
            val /= global_sum;
        }
    }

    // Synchronize a after softmax
    for (size_t i = 0; i < a.size(); i++) {
        vector<double> local_row = a[i];
        vector<double> global_row(local_row.size(), 0.0);
        
        MPI_Allreduce(local_row.data(), global_row.data(), local_row.size(), 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Since we've applied softmax already to our rows, divide by size
        // (except for rows that only one process handled)
        if (i >= start_row && i < end_row) {
            for (size_t j = 0; j < global_row.size(); j++) {
                global_row[j] /= size;
            }
        }
        
        a[i] = global_row;
    }
}

vector<vector<double>> softmax(vector<vector<double>>& a, int rank, int size) {
    softmax_rows(a, rank, size);
    return a;
}

vector<vector<double>> matrixTranspose(vector<vector<double>>& a, int rank, int size) {
    if (a.empty() || a[0].empty()) {
        return {};
    }
    
    vector<vector<double>> b(a[0].size(), vector<double>(a.size(), 0.0));
    
    // Calculate work distribution
    size_t rows_per_process = (a.size() + size - 1) / size; // Ceiling division
    size_t start_row = rank * rows_per_process;
    size_t end_row = min(start_row + rows_per_process, a.size());

    for (size_t i = start_row; i < end_row; i++) {
        for (size_t j = 0; j < a[i].size(); j++) {
            b[j][i] = a[i][j];
        }
    }

    // Synchronize b after transpose
    for (size_t i = 0; i < b.size(); i++) {
        vector<double> local_row = b[i];
        vector<double> global_row(local_row.size(), 0.0);
        
        MPI_Allreduce(local_row.data(), global_row.data(), local_row.size(), 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        b[i] = global_row;
    }

    return b;
}

vector<vector<double>> read_data(string filename, size_t sequence_length, size_t embed_dim, int rank, int size) {
    vector<vector<double>> data;
    size_t data_size = 0;

    // Only rank 0 reads the file
    if (rank == 0) {
        ifstream file(filename);
        if (!file.is_open()) {
            cout << "File not found!" << endl;
            return {};
        }
        
        string line;
        while (getline(file, line)) {
            stringstream ss(line);
            vector<double> vec(embed_dim, 0.0);
            for (size_t i = 0; i < embed_dim; i++) {
                if (!(ss >> vec[i])) {
                    break;
                }
            }
            if (vec.size() != embed_dim) {
                cout << "Mismatch in sample size. Skipping sample" << endl;
                continue;
            }
            data.push_back(vec);
        }
        data_size = data.size();
    }

    // Broadcast data size
    MPI_Bcast(&data_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    if (data_size == 0) {
        if (rank == 0) {
            cout << "No valid data read from file!" << endl;
        }
        return {};
    }

    // Allocate space for data on all processes
    if (rank != 0) {
        data.resize(data_size, vector<double>(embed_dim, 0.0));
    }

    // Broadcast data row by row to avoid large buffer issues
    for (size_t i = 0; i < data_size; i++) {
        MPI_Bcast(data[i].data(), embed_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    return data;
}

vector<vector<double>> get_positional_encoding(size_t sequence_length, size_t d_model, int rank, int size) {
    vector<vector<double>> positional_encodings(sequence_length, vector<double>(d_model, 0.0f));
    
    // Calculate work distribution
    size_t rows_per_process = (sequence_length + size - 1) / size; // Ceiling division
    size_t start_row = rank * rows_per_process;
    size_t end_row = min(start_row + rows_per_process, sequence_length);

    for (size_t pos = start_row; pos < end_row; pos++) {
        for (size_t i = 0; i < d_model; i++) {
            if (i % 2 == 0) {
                positional_encodings[pos][i] = sin(pos / pow(10000, (double)i / d_model));
            } else {
                positional_encodings[pos][i] = cos(pos / pow(10000, (double)(i - 1) / d_model));
            }
        }
    }

    // Synchronize positional_encodings across all processes
    for (size_t i = 0; i < sequence_length; i++) {
        vector<double> local_row = positional_encodings[i];
        vector<double> global_row(d_model, 0.0);
        
        MPI_Allreduce(local_row.data(), global_row.data(), d_model, 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        positional_encodings[i] = global_row;
    }

    return positional_encodings;
}

vector<vector<double>> add_vectors(vector<vector<double>>& a, vector<vector<double>>& b, int rank, int size) {
    size_t seq_len = a.size();
    size_t dim = a[0].size();
    vector<vector<double>> result(seq_len, vector<double>(dim, 0.0));
    
    // Calculate work distribution
    size_t rows_per_process = (seq_len + size - 1) / size; // Ceiling division
    size_t start_row = rank * rows_per_process;
    size_t end_row = min(start_row + rows_per_process, seq_len);
    
    // Add vectors locally
    for (size_t i = start_row; i < end_row; i++) {
        for (size_t j = 0; j < dim; j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    
    // Synchronize result
    syncMatrix(result, rank, size);
    
    return result;
}


vector<vector<double>> layer_norm(vector<vector<double>>& input, vector<double>& gamma, vector<double>& beta, float epsilon, int rank, int size) {
    size_t seq_len = input.size();
    size_t dim = input[0].size();
    vector<vector<double>> output(seq_len, vector<double>(dim, 0.0f));
    
    // Calculate work distribution
    size_t rows_per_process = (seq_len + size - 1) / size; // Ceiling division
    size_t start_row = rank * rows_per_process;
    size_t end_row = min(start_row + rows_per_process, seq_len);

    for (size_t i = start_row; i < end_row; ++i) {
        // Compute mean
        double mean = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            mean += input[i][j];
        }
        mean /= dim;

        // Compute variance
        double var = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            var += (input[i][j] - mean) * (input[i][j] - mean);
        }
        var /= dim;

        // Apply normalization with gamma and beta
        for (size_t j = 0; j < dim; ++j) {
            output[i][j] = gamma[j] * ((input[i][j] - mean) / sqrt(var + epsilon)) + beta[j];
        }
    }

    // Synchronize output
    syncMatrix(output, rank, size);
    
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
    
        vector<vector<double>> forward(vector<vector<double>>& x, int rank, int size) {
            vector<vector<double>> attn_output = mha.forward(x, rank, size);
            vector<vector<double>> addLayer1 = add_vectors(x, attn_output, rank, size);
            vector<vector<double>> norm1 = layer_norm(addLayer1, gamma, beta, 1e-6, rank, size);
            vector<vector<double>> ff_output = ff.forward(norm1, rank, size);
            vector<vector<double>> addLayer2 = add_vectors(norm1, ff_output, rank, size);
            vector<vector<double>> norm2 = layer_norm(addLayer2, gamma, beta, 1e-6, rank, size);
            return norm2;
        }
    };


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Transformer parameters
    size_t d_model = 500;
    size_t embed_dim = 500;
    size_t sequence_length = 100;
    size_t num_heads = 50;
    size_t ff_dim = 1280;
    string input_file = "dataset_vectors.txt";

    if (rank == 0) {
        cout << "Loading the data from the file..." << endl;
    }

    vector<vector<double>> input_vectors = read_data(input_file, sequence_length, embed_dim, rank, size);

    if (input_vectors.empty()) {
        if (rank == 0) {
            cout << "No data found in the file. Exiting..." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        cout << "Data read successfully with sequence length " << input_vectors.size() << endl;
    }

    double start_time = MPI_Wtime();

    // Select the first sample
    vector<double> sample;
    if (input_vectors.size() > 0) {
        sample = input_vectors[0];
    } else {
        if (rank == 0) {
            cout << "Empty input vectors" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Create and fill sample_vector
    vector<vector<double>> sample_vector(sequence_length, vector<double>(embed_dim, 0.0f));
    
    // Calculate work distribution
    size_t rows_per_process = (sequence_length + size - 1) / size; // Ceiling division
    size_t start_row = rank * rows_per_process;
    size_t end_row = min(start_row + rows_per_process, sequence_length);
    
    // Fill sample_vector with repeated sample data
    for (size_t i = start_row; i < end_row; i++) {
        for (size_t j = 0; j < embed_dim; j++) {
            sample_vector[i][j] = sample[j % sample.size()];
        }
    }
    
    // Synchronize sample_vector
    syncMatrix(sample_vector, rank, size);
    
    // Create encoder layer and process
    EncoderLayer encoder_layer(d_model, num_heads, ff_dim);
    vector<vector<double>> encoder_output = encoder_layer.forward(sample_vector, rank, size);

    // Output results
    if (rank == 0) {
        cout << "Output after input data is passed through the transformer : " << endl;
        ofstream outputFile("output.txt");
        
        for (size_t i = 0; i < min(size_t(5), encoder_output.size()); ++i) {
            for (size_t j = 0; j < min(size_t(10), encoder_output[i].size()); ++j) {
                outputFile << encoder_output[i][j] << " ";
                cout << encoder_output[i][j] << " ";
            }
            cout << "..." << endl;
        }
        cout << "..." << endl;
        outputFile.close();
    }

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    if (rank == 0) {
        cout << "Processes: " << size << " | Time: " << elapsed_time << " seconds.\n";
        ofstream output_file("execution_times.txt", ios::app);
        output_file << size << " " << elapsed_time << "\n";
        output_file.close();
    }

    MPI_Finalize();
    return 0;
}