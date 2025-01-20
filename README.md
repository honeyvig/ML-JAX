# ML-JAX
JAX is a powerful numerical computing library that allows for high-performance machine learning research. It provides automatic differentiation (autograd) and is highly optimized for hardware acceleration (GPU/TPU support). Here's a simple Python example using JAX that demonstrates how to use its features for basic numerical tasks:
Example 1: Basic Operations and Automatic Differentiation

import jax
import jax.numpy as jnp

# Define a simple function
def f(x):
    return x**2 + 2*x + 1

# Create a JAX array
x = jnp.array([1.0, 2.0, 3.0])

# Apply the function to the array
y = f(x)

print("Function output:", y)

# Compute the gradient (derivative) of the function
grad_f = jax.grad(f)

# Compute the gradient at a specific point
gradient_at_x = grad_f(x)
print("Gradient of f at x:", gradient_at_x)

Explanation:

    jax.numpy provides a version of NumPy that is compatible with JAX's features like automatic differentiation and hardware acceleration.
    jax.grad automatically computes the gradient of a function with respect to its input.
    jnp.array is used to create JAX arrays, which are similar to NumPy arrays but are designed to support JAX's optimizations.

Example 2: Using JAX for Matrix Multiplication

import jax
import jax.numpy as jnp

# Define two matrices
A = jnp.array([[1, 2], [3, 4]])
B = jnp.array([[5, 6], [7, 8]])

# Perform matrix multiplication
C = jnp.dot(A, B)
print("Matrix multiplication result (C = A * B):")
print(C)

# Compute the gradient of the result with respect to matrix A
grad_A = jax.grad(lambda A: jnp.sum(jnp.dot(A, B)))

# Gradient of the result with respect to A
gradient_at_A = grad_A(A)
print("Gradient of the result with respect to A:")
print(gradient_at_A)

Explanation:

    We perform matrix multiplication using jax.numpy.dot and calculate its gradient with respect to matrix A using jax.grad.
    JAX allows easy automatic differentiation of matrix operations.

Example 3: Using JAX with GPU/TPU

If you have access to a GPU or TPU, JAX automatically uses it for computation if the device is available. To explicitly move data to a GPU or TPU, you can use jax.device_put.

import jax
import jax.numpy as jnp

# Create a large array
x = jnp.ones((1000, 1000))

# Move array to GPU (if available)
x_device = jax.device_put(x)

# Perform matrix multiplication on the device
y = jnp.dot(x_device, x_device.T)

print("Matrix multiplication completed on GPU/TPU")

Explanation:

    This code moves the array to the GPU/TPU using jax.device_put. JAX automatically handles the device-specific operations.
    JAX operations are automatically parallelized when using GPUs/TPUs.

Example 4: JAX for Optimization

JAX is also widely used for optimization tasks. Here is a simple example using JAX to minimize a function:

import jax
import jax.numpy as jnp
from jax import grad, jit

# Define a simple quadratic function
def loss_function(w):
    return jnp.sum(w**2)

# Compute the gradient of the loss function
grad_loss = grad(loss_function)

# Gradient descent optimizer
def update(w, lr=0.1):
    return w - lr * grad_loss(w)

# Initialize parameters
w = jnp.array([2.0, 3.0])

# Perform a few steps of gradient descent
for i in range(10):
    w = update(w)
    print(f"Step {i+1}: w = {w}, Loss = {loss_function(w)}")

Explanation:

    We define a simple loss function (loss_function) and use jax.grad to compute its gradient.
    The gradient is used in a basic gradient descent loop to minimize the function.

Example 5: JAX with JIT Compilation for Performance

JAX provides Just-In-Time (JIT) compilation, which speeds up computations by compiling functions ahead of time.

import jax
import jax.numpy as jnp
from jax import jit

# Define a simple function
def f(x):
    return jnp.sin(x) ** 2 + jnp.cos(x) ** 2

# JIT compile the function
f_jit = jit(f)

# Test the JIT compiled function
x = jnp.array([1.0, 2.0, 3.0])
print(f"JIT-compiled function result: {f_jit(x)}")

Explanation:

    The jit decorator compiles the function to make it faster for repeated calls.
    JAX handles this compilation automatically to ensure better performance, especially in loops or large-scale computations.

Conclusion:

JAX is a very powerful library for numerical computation and machine learning tasks, with its focus on automatic differentiation, hardware acceleration, and JIT compilation. By utilizing JAX's neural network capabilities, you can optimize machine learning models and other numerical tasks efficiently.

These are just some basic examples to get started. JAX has a broad set of capabilities for high-performance computing, especially for deep learning and scientific computing tasks.
