#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

vector<double> read_input() {
    FILE *fp = fopen("random_doubles_numpy.txt", "r");
    if (fp == NULL) {
        cout << "Error opening file" << endl;
        exit(1);
    }
    vector<double> a;
    double x;
    while (fscanf(fp, "%lf", &x) == 1) {
        a.push_back(x);
    }
    fclose(fp);
    return a;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <num_threads>" << endl;
        return 1;
    }

    int num_threads = atoi(argv[1]);
    omp_set_num_threads(num_threads);

    vector<double> a = read_input();
    cout << "The size of input array is : " << a.size() << endl;

    double sum = 0;
    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        double local_sum = 0;
        #pragma omp for
        for (size_t i = 0; i < a.size(); i++) {
            local_sum += a[i];
        }
        #pragma omp critical
        sum += local_sum;
    }
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    cout << "Threads: " << num_threads << ", Sum: " << sum << ", Time: " << elapsed_time << " seconds" << endl;

    return 0;
}
