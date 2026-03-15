#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <complex>
#include <limits>
#include <utility>     

using namespace std;

using Complex = complex<double>;
const double PI = acos(-1.0);

class Matrix {
private:
    vector<vector<double>> data;
    size_t rows_, cols_;

public:
    Matrix(size_t r = 0, size_t c = 0, double val = 0.0)
        : rows_(r), cols_(c), data(r, vector<double>(c, val)) {}

    Matrix(const vector<vector<double>>& d)
        : data(d), rows_(d.size()), cols_(d.empty() ? 0 : d[0].size()) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    double& operator()(size_t i, size_t j)       { return data[i][j]; }
    const double& operator()(size_t i, size_t j) const { return data[i][j]; }

    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) throw runtime_error("Dimension mismatch in matrix multiplication");
        Matrix res(rows_, other.cols_, 0.0);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < other.cols_; ++j)
                for (size_t k = 0; k < cols_; ++k)
                    res(i,j) += (*this)(i,k) * other(k,j);
        return res;
    }

    void print(int prec = 6) const {
        cout << fixed << setprecision(prec);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j)
                cout << setw(12) << (*this)(i,j);
            cout << '\n';
        }
    }
};

double determinant(Matrix A) {
    if (A.rows() != A.cols()) throw runtime_error("Square matrix required");
    size_t n = A.rows();
    double det = 1.0;
    for (size_t i = 0; i < n; ++i) {
        size_t pivot = i;
        for (size_t j = i+1; j < n; ++j)
            if (abs(A(j,i)) > abs(A(pivot,i))) pivot = j;
        if (pivot != i) {
            for (size_t j = 0; j < n; ++j) swap(A(i,j), A(pivot,j));
            det = -det;
        }
        if (abs(A(i,i)) < 1e-12) return 0.0;
        det *= A(i,i);
        for (size_t j = i+1; j < n; ++j) {
            double f = A(j,i) / A(i,i);
            for (size_t k = i; k < n; ++k) A(j,k) -= f * A(i,k);
        }
    }
    return det;
}

vector<double> solve(const Matrix& A, const vector<double>& b) {
    size_t n = A.rows();
    if (n != A.cols() || n != b.size()) throw runtime_error("Invalid dimensions");
    Matrix aug(n, n+1);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) aug(i,j) = A(i,j);
        aug(i,n) = b[i];
    }
    for (size_t i = 0; i < n; ++i) {
        size_t p = i;
        for (size_t j = i+1; j < n; ++j)
            if (abs(aug(j,i)) > abs(aug(p,i))) p = j;
        if (p != i) for (size_t j = 0; j <= n; ++j) swap(aug(i,j), aug(p,j));
        if (abs(aug(i,i)) < 1e-12) throw runtime_error("Singular matrix");
        for (size_t j = i+1; j < n; ++j) {
            double f = aug(j,i) / aug(i,i);
            for (size_t k = i; k <= n; ++k) aug(j,k) -= f * aug(i,k);
        }
    }
    vector<double> x(n);
    for (int i = n-1; i >= 0; --i) {
        x[i] = aug(i,n);
        for (size_t j = i+1; j < n; ++j) x[i] -= aug(i,j) * x[j];
        x[i] /= aug(i,i);
    }
    return x;
}

template<typename Func>
double trapezoidal(Func&& f, double a, double b, int n = 10000) {
    double h = (b - a) / n;
    double s = 0.5 * (f(a) + f(b));
    for (int i = 1; i < n; ++i) s += f(a + i * h);
    return s * h;
}

template<typename Func>
double simpson(Func&& f, double a, double b, int n = 10000) {
    if (n % 2 == 1) ++n;
    double h = (b - a) / n;
    double s = f(a) + f(b);
    for (int i = 1; i < n; ++i)
        s += (i % 2 == 0 ? 2.0 : 4.0) * f(a + i * h);
    return s * h / 3.0;
}

template<typename Func>
double bisection(Func&& f, double a, double b, double tol = 1e-10, int maxit = 1000) {
    if (f(a) * f(b) >= 0) throw runtime_error("No sign change in interval");
    for (int it = 0; it < maxit; ++it) {
        double c = (a + b) / 2;
        if (abs(f(c)) < tol || (b - a) < tol) return c;
        if (f(a) * f(c) < 0) b = c; else a = c;
    }
    return (a + b) / 2;
}

template<typename Func>
vector<pair<double,double>> rk4(Func&& f, double x0, double y0, double xend, int steps) {
    double h = (xend - x0) / steps;
    vector<pair<double,double>> sol{{x0, y0}};
    double x = x0, y = y0;
    for (int i = 0; i < steps; ++i) {
        double k1 = f(x, y);
        double k2 = f(x + h/2, y + h*k1/2);
        double k3 = f(x + h/2, y + h*k2/2);
        double k4 = f(x + h,     y + h*k3);
        y += h/6 * (k1 + 2*k2 + 2*k3 + k4);
        x += h;
        sol.emplace_back(x, y);
    }
    return sol;
}

size_t next_pow2(size_t n) {
    size_t p = 1; while (p < n) p <<= 1; return p;
}

void fft(vector<Complex>& a, bool inverse) {
    size_t n = a.size();
    if ((n & (n-1)) != 0) throw runtime_error("FFT size must be power of 2");

    // bit reversal
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) swap(a[i], a[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = 2*PI/len * (inverse ? 1 : -1);
        Complex wl(cos(ang), sin(ang));
        for (size_t i = 0; i < n; i += len) {
            Complex w(1);
            for (size_t j = 0; j < len/2; ++j) {
                Complex u = a[i+j], v = a[i+j+len/2] * w;
                a[i+j]     = u + v;
                a[i+j+len/2] = u - v;
                w *= wl;
            }
        }
    }
    if (inverse) for (auto& z : a) z /= double(n);
}

vector<Complex> fft_forward(const vector<double>& real_in) {
    size_t n = next_pow2(real_in.size());
    vector<Complex> z(n, 0.0);
    for (size_t i = 0; i < real_in.size(); ++i) z[i] = real_in[i];
    fft(z, false);
    return z;
}

Matrix read_matrix(size_t n) {
    Matrix m(n, n);
    cout << "Enter " << n << "×" << n << " matrix row by row:\n";
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            cout << "A[" << i << "][" << j << "] = ";
            cin >> m(i,j);
        }
    return m;
}

vector<double> read_vector(size_t n) {
    vector<double> v(n);
    cout << "Enter vector of size " << n << ":\n";
    for (size_t i = 0; i < n; ++i) {
        cout << "b[" << i << "] = ";
        cin >> v[i];
    }
    return v;
}

int main() {
    cout << fixed << setprecision(8);
    cout << "=======================================\n";
    cout << "      Numerical Methods Toolkit\n";
    cout << "      Interactive mode\n";
    cout << "=======================================\n";

    while (true) {
        cout << "\nChoose category:\n"
             << "  1) Linear Algebra\n"
             << "  2) Numerical Integration\n"
             << "  3) Root Finding\n"
             << "  4) ODE Solving (RK4)\n"
             << "  5) FFT (forward transform)\n"
             << "  0) Exit\n> ";

        int cat;
        if (!(cin >> cat)) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Please enter a number.\n";
            continue;
        }

        if (cat == 0) break;

        try {
            switch (cat) {
            case 1: { 
                cout << "\n  a) Solve Ax = b\n"
                     << "  b) Compute determinant\n"
                     << "  c) Matrix × Matrix\n> ";
                char sub; cin >> sub;

                cout << "Matrix size n = "; size_t n; cin >> n;
                Matrix A = read_matrix(n);

                if (sub == 'a' || sub == 'A') {
                    auto b = read_vector(n);
                    auto x = solve(A, b);
                    cout << "\nSolution x:\n";
                    for (size_t i = 0; i < x.size(); ++i)
                        cout << "x[" << i << "] = " << x[i] << '\n';
                }
                else if (sub == 'b' || sub == 'B') {
                    cout << "\ndet(A) = " << determinant(A) << '\n';
                }
                else if (sub == 'c' || sub == 'C') {
                    cout << "Second matrix columns m = "; size_t m; cin >> m;
                    Matrix B(n, m);
                    cout << "Enter second matrix:\n";
                    for (size_t i = 0; i < n; ++i)
                        for (size_t j = 0; j < m; ++j)
                            cin >> B(i,j);
                    cout << "\nA × B =\n";
                    (A * B).print();
                }
                break;
            }

            case 2: { 
                cout << "\n  a) Trapezoidal rule\n  b) Simpson's rule\n> ";
                char method; cin >> method;

                cout << "Interval a b: "; double a, b; cin >> a >> b;

                string func_str;
                cout << "Function (limited support): sin(x), cos(x), x*x, exp(-x)\n> ";
                cin >> func_str;

                auto f = [&](double x) -> double {
                    if (func_str == "sin(x)") return sin(x);
                    if (func_str == "cos(x)") return cos(x);
                    if (func_str == "x*x")    return x*x;
                    if (func_str == "exp(-x)") return exp(-x);
                    throw runtime_error("Unsupported function (only sin(x), cos(x), x*x, exp(-x) for now)");
                };

                double result = (method == 'a' || method == 'A')
                              ? trapezoidal(f, a, b)
                              : simpson(f, a, b);

                cout << "\nApproximate integral = " << result << '\n';
                break;
            }

            case 3: {
                cout << "Interval a b: "; double a, b; cin >> a >> b;

                string func_str;
                cout << "Function (limited): x*x-2, sin(x)-0.5\n> ";
                cin >> func_str;

                auto f = [&](double x) {
                    if (func_str == "x*x-2")   return x*x - 2;
                    if (func_str == "sin(x)-0.5") return sin(x) - 0.5;
                    throw runtime_error("Unsupported function");
                };

                cout << "\nApproximate root = " << bisection(f, a, b) << '\n';
                break;
            }

            case 4: { 
                cout << "Equation y' = f(x,y)   examples: -y , y*(1-y) , -2*x*y\n> ";
                string ode_str; cin.ignore(); getline(cin, ode_str);

                auto f = [&](double x, double y) -> double {
                    if (ode_str == "-y")         return -y;
                    if (ode_str == "y*(1-y)")    return y * (1 - y);
                    if (ode_str == "-2*x*y")     return -2 * x * y;
                    throw runtime_error("Unsupported ODE right-hand side");
                };

                cout << "x0 = "; double x0; cin >> x0;
                cout << "y0 = "; double y0; cin >> y0;
                cout << "x_end = "; double xend; cin >> xend;
                cout << "Number of steps = "; int steps; cin >> steps;

                auto solution = rk4(f, x0, y0, xend, steps);

                cout << "\n   x          y\n";
                for (const auto& [x, y] : solution) {
                    cout << x << "   " << y << '\n';
                }
                break;
            }

            case 5: { 
                cout << "Number of samples: "; size_t n; cin >> n;
                vector<double> signal(n);
                cout << "Enter " << n << " values:\n";
                for (size_t i = 0; i < n; ++i) {
                    cout << "x[" << i << "] = ";
                    cin >> signal[i];
                }

                auto spectrum = fft_forward(signal);
                cout << "\nMagnitude spectrum (first min(10, N) bins):\n";
                size_t show = min<size_t>(10, spectrum.size());
                for (size_t i = 0; i < show; ++i) {
                    cout << "bin " << i << ": |X| = " << abs(spectrum[i]) << '\n';
                }
                break;
            }

            default:
                cout << "Invalid choice.\n";
            }
        }
        catch (const exception& e) {
            cerr << "\nError: " << e.what() << '\n';
        }

        cout << "\nPress Enter to continue...";
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        string dummy; getline(cin, dummy);
    }

    cout << "Goodbye.\n";
    return 0;
}