#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>

using namespace std;

const int N = 10000; 

void generateMatrix(const string &filename) {
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "Error: Unable to open file " << filename << " for writing." << endl;
        exit(1);
    }

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 1000.0);

    outfile << fixed << setprecision(6);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            outfile << dis(gen) << " ";
        }
        outfile << "\n";
    }

    outfile.close();
    cout << "Generated matrix saved to " << filename << endl;
}

int main() {
    generateMatrix("matrix1.txt");
    generateMatrix("matrix2.txt");
    return 0;
}
