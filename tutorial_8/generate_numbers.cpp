#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>  
using namespace std;

int main() {
    const int N = 10000000;  
    ofstream outfile("data.txt");

    if (!outfile.is_open()) {
        cerr << "Error: Unable to open file for writing." << endl;
        return 1;
    }

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<double> dis(0.0, 1000.0);

    outfile << fixed << setprecision(6);

    for (int i = 0; i < N; ++i) {
        double number = dis(gen);
        outfile << number << "\n";
    }

    outfile.close();
    cout << "Generated " << N << " double-precision numbers in data.txt with at least 6 decimal digits." << endl;
    return 0;
}
