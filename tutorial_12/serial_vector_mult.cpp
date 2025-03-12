#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int N = 10000;

void readMatrix(const string &filename, vector<vector<double>> &matrix) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        exit(1);
    }

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            infile >> matrix[i][j];

    infile.close();
}

void writeMatrix(const string &filename, const vector<vector<double>> &matrix) {
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "Error: Unable to open file " << filename << " for writing." << endl;
        exit(1);
    }

    for (const auto &row : matrix) {
        for (double val : row)
            outfile << val << " ";
        outfile << "\n";
    }

    outfile.close();
}

int main() {
    vector<vector<double>> A(N, vector<double>(N));
    vector<vector<double>> B(N, vector<double>(N));
    vector<vector<double>> C(N, vector<double>(N));

    readMatrix("matrix1.txt", A);
    readMatrix("matrix2.txt", B);

    auto start = high_resolution_clock::now();

    // Serial matrix addition
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            C[i][j] = A[i][j] * B[i][j];

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    writeMatrix("result_serial.txt", C);

    cout << "Serial Matrix Multiplication Time: " << duration.count() << " ms" << endl;
    return 0;
}
