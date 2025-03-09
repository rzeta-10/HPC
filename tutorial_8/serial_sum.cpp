#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <vector>
using namespace std;

int main() {
    ifstream infile("data.txt");
    vector<double> numbers;
    double x;

    while (infile >> x) {
        numbers.push_back(x);
    }

    double sum = 0.0;
    
    auto start = chrono::high_resolution_clock::now();

    for (double num : numbers) {
        sum += num;
    }
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Serial Sum = " << fixed << setprecision(6) << sum << endl;
    cout << "Serial Time = " << elapsed.count() << " seconds" << endl;
    
    return 0;
}