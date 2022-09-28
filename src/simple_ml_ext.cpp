#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void print_matrix(const float *X, size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i)
    {
       for (size_t j = 0; j < n; ++j)
       {
          std::cout << X[i * n + j] << " ";
       }

       // Newline for new row
       std::cout << std::endl;
    }
}

void multiply_by_const(
    float *X,
    float value,
    size_t m,
    size_t n
) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            X[i * n + j] *= value;
        }
    }
}

void subtract(
    float *X,
    const float *Z,
    size_t m,
    size_t n
) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            X[i * n + j] -= Z[i * n + j];
        }
    }
}

void transpose(
    const float *X,
    size_t m,
    size_t n,
    float *res
) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            res[j * m + i] = X[i * n + j];
        }
    }
}

void onehot(
    const unsigned char *y,
    size_t m,
    size_t k,
    float *res
) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            res[i * k + j] = (y[i] == j) ? 1 : 0;
        }
    }
}

void dot(
     const float *X,
     const float *Z,
     size_t m,
     size_t n,
     size_t k,
     float *res
) {
     for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l < n; ++l) {
                res[i * k + j] += static_cast<double>(X[i * n + l]) * Z[l * k + j];
            }
        }
    }
}

void softmax(
    float* Z,
    size_t m,
    size_t k
) {
	for (size_t i = 0; i < m; ++i) {
        float exp_sum = 0;
        for (size_t j = 0; j < k; ++j) {
             Z[i * k + j] = exp(Z[i * k + j]);
             exp_sum += Z[i * k + j];
        }
        for (size_t j = 0; j < k; ++j) {
             Z[i * k + j] /= exp_sum;
        }
    }
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    float lr_step = lr / static_cast<float>(batch);

    float* onehot_y = new float[m * k]();
    onehot(y, m, k, onehot_y);

    for (size_t i = 0; i < m; i += batch) {
        float* softmax_res = new float[batch * k]();
        dot(X + (i * n), theta, batch, n, k, softmax_res);
        softmax(softmax_res, batch, k);
        // print_matrix(softmax_res, batch, k);
        subtract(softmax_res, onehot_y + (i * k), batch, k);

        float *transposed_batch = new float[n * batch]();
        transpose(X + (i * n), batch, n, transposed_batch);

        float* gradient = new float[n * k]();
        dot(transposed_batch, softmax_res, n, batch, k, gradient);
        multiply_by_const(gradient, lr_step, n, k);
        subtract(theta, gradient, n, k);

        delete[] softmax_res;
        delete[] transposed_batch;
        delete[] gradient;

    }
    delete[] onehot_y;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
