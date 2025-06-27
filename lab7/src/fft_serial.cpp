#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

const double PI = acos(-1.0);

// 递归实现的 FFT
void fft(std::vector<std::complex<double> >& a, bool invert) {
    int n = a.size();
    if (n <= 1) return;

    std::vector<std::complex<double> > a0(n / 2), a1(n / 2);
    for (int i = 0, j = 0; i < n; i += 2, ++j) {
        a0[j] = a[i];
        a1[j] = a[i + 1];
    }

    fft(a0, invert);
    fft(a1, invert);

    double ang = 2 * PI / n * (invert ? -1 : 1);
    std::complex<double> w(1), wn(cos(ang), sin(ang));

    for (int i = 0; i < n / 2; ++i) {
        std::complex<double> t = w * a1[i];
        a[i] = a0[i] + t;
        a[i + n / 2] = a0[i] - t;
        w *= wn;
    }

    if (invert) {
        for (int i = 0; i < n; ++i) {
            a[i] /= 2;
        }
    }
}

int main(int argc, char* argv[]) {
    int n = 8; // 默认输入大小
    if (argc > 1) {
        n = std::atoi(argv[1]);
        if (n <= 0 || (n & (n - 1)) != 0) { // 必须是2的幂
            std::cerr << "Input size n must be a power of 2." << std::endl;
            return 1;
        }
    }

    std::vector<std::complex<double> > data(n);
    std::cout << "Input data (size " << n << "):" << std::endl;
    for (int i = 0; i < n; ++i) {
        data[i] = std::complex<double>(rand() % 10, 0); // 简单生成一些实数数据
        // data[i] = std::complex<double>(i, 0);
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    std::vector<std::complex<double> > data_fft = data;
    fft(data_fft, false);

    std::cout << "FFT result:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << data_fft[i] << " ";
    }
    std::cout << std::endl;

    std::vector<std::complex<double> > data_ifft = data_fft;
    fft(data_ifft, true);

    std::cout << "IFFT result (should be close to input):" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << data_ifft[i] << " ";
    }
    std::cout << std::endl;

    return 0;
} 