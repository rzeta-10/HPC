#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>

using namespace std;

const size_t kNumCount = 10'000'000;

void GenerateAndSave(const string& filename) {
    ofstream file(filename);
    if (!file) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    random_device rd;
    mt19937_64 gen(rd());
    uniform_real_distribution<double> dist(0.0, 10'000.0);

    file << scientific << setprecision(10);

    for (size_t i = 0; i < kNumCount; ++i) {
        file << dist(gen) << "\n";
    }

    file.close();
    cout << "Successfully written " << kNumCount << " double values to " << filename << endl;
}

int main() {
    GenerateAndSave("dataset1.txt");
    GenerateAndSave("dataset2.txt");

    return 0;
}