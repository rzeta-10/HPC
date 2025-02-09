#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

vector<double> read_input(const string& filename) {
    FILE *fp = fopen(filename.c_str(), "r");
    if (fp == NULL) {
        cout << "Error opening file: " << filename << endl;
        exit(1);
    }
    vector<double> vec;
    double x;
    while (fscanf(fp, "%lf", &x) == 1) {
        vec.push_back(x);
    }
    fclose(fp);
    return vec;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <num_threads>" << endl;
        return 1;
    }

    int num_threads = atoi(argv[1]);
    omp_set_num_threads(num_threads);

    // Read input vectors
    vector<double> A = read_input("random_doubles_numpy.txt");
    vector<double> B = read_input("random_doubles_numpy.txt");  // Using the same file for testing

    size_t N = A.size();
    if (B.size() != N) {
        cout << "Error: Vector sizes do not match!" << endl;
        return 1;
    }

    vector<double> C(N, 0.0);  // Result vector

    double start_time = omp_get_wtime();
    double ans = 0;
    #pragma omp parallel
    {
        double local_sum = 0;
        #pragma omp for
        for (size_t i = 0; i < N; i++) {
            local_sum += A[i] * B[i];
        }
        #pragma omp critical
        ans += local_sum;
    }

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    cout << "Threads: " << num_threads << ", Sum: " << ans << ", Time: " << elapsed_time << " seconds" << endl;

    return 0;
}
