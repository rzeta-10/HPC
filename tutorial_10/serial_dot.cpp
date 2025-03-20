#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

using namespace std;

#define SIZE 10000000

int main() {
    ifstream input_file1("dataset1.txt");
    ifstream input_file2("dataset2.txt");
    vector<double> data_vector1(SIZE);
    vector<double> data_vector2(SIZE);
    double result_dot_product = 0.0;

    if (!input_file1 || !input_file2) {
        cerr << "Error opening files" << endl;
        return 1;
    }

    for (size_t i = 0; i < SIZE; ++i) {
        if (!(input_file1 >> data_vector1[i]) || !(input_file2 >> data_vector2[i])) {
            cerr << "Error reading numbers from files" << endl;
            return 1;
        }
    }

    input_file1.close();
    input_file2.close();

    clock_t start_time = clock();

    for (size_t i = 0; i < SIZE; ++i) {
        result_dot_product += data_vector1[i] * data_vector2[i];
    }

    clock_t end_time = clock();

    cout << "Dot product: " << result_dot_product << std::endl;
    cout << "Time taken: " << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC << " seconds" << endl;

    return 0;
}