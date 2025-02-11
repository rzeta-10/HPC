#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <num_threads>\n";
        return 1;
    }

    int num_threads = stoi(argv[1]);  // Get number of threads from command-line argument
    omp_set_num_threads(num_threads);

    int N = 10000;  // Matrix size

    // Allocate memory for matrices
    vector<vector<double>> matrix1(N, vector<double>(N));
    vector<vector<double>> matrix2(N, vector<double>(N));
    vector<vector<double>> result(N, vector<double>(N));

    // Open file for reading
    ifstream file("matrix.txt");
    if (!file) {
        cerr << "Error: Unable to open matrix.txt\n";
        return 1;
    }

    cout << "Reading matrix from file...\n";

    // Read matrix1 from file
    string line;
    for (int i = 0; i < N; i++) {
        if (!getline(file, line)) {
            cerr << "Error: Unexpected end of file\n";
            return 1;
        }
        stringstream ss(line);
        for (int j = 0; j < N; j++) {
            ss >> matrix1[i][j];
        }
    }
    cout << "Matrix 1 loaded successfully!\n";

    // Reset file stream to read again
    file.clear();
    file.seekg(0, ios::beg);

    cout << "Reading matrix again for Matrix 2...\n";

    // Read matrix2 from file
    for (int i = 0; i < N; i++) {
        if (!getline(file, line)) {
            cerr << "Error: Unexpected end of file\n";
            return 1;
        }
        stringstream ss(line);
        for (int j = 0; j < N; j++) {
            ss >> matrix2[i][j];
        }
    }
    file.close();
    cout << "Matrix 2 loaded successfully!\n";
    cout << matrix1.size() << " " << matrix1[0].size() << endl;
    cout << matrix2.size() << " " << matrix2[0].size() << endl;

    // Perform matrix addition
    double start_time = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    cout << "Threads: " << num_threads << " | Time: " << elapsed_time << " seconds.\n";

    // Save execution time to a file
    ofstream output_file("execution_times.txt", ios::app);
    output_file << num_threads << " " << elapsed_time << "\n";
    output_file.close();

    return 0;
}
