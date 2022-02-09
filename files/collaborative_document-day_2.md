![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document. Day 2, February 9

2022-02-08-ds-gpu

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](https://hackmd.io/@isazi/Bk3WuTDAF)

Collaborative Document day 1: [link](https://hackmd.io/@isazi/rk-rL6vRY)

Collaborative Document day 2: [link](https://hackmd.io/@isazi/Bk3WuTDAF)

## 👮Code of Conduct

* Participants are expected to follow those guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, type `/hand` in the chat window.

To get help, type `/help` in the chat window.

You can ask questions in the document or chat window and helpers will try to help you.

## 🖥 Workshop website

* [Course](https://esciencecenter-digital-skills.github.io/2022-02-08-ds-gpu/)
* [JupyterHub](https://jupyter.lisa.surfsara.nl/jhlsrf012)
* [Google Colab](https://colab.research.google.com)
* [Post-workshop Survey](https://www.surveymonkey.com/r/78GNSJB)

## 👩‍🏫👩‍💻🎓 Instructors

Alessio Sclocco, Hanno Spreeuw, Johan Hidding

## 🧑‍🙋 Helpers

Lieke de Boer

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call

Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city

## 🗓️ Agenda

| time  | what |
|-------|---|
| 09:00 | Welcome and icebreaker |
| 09:15 | Introduction to CUDA |
| 10:15 | Coffee break |
| 10:30 | CUDA memories and their use |
| 11:30 | Coffee break |
| 11:45 | Data sharing and synchronization |
| 12:45 | Wrap-up and post-workshop survey |
| 13:00 | END |

## 🔧 Exercises

### Challenge: Loose threads

We know enough now to pause for a moment and do a little exercise. Assume that in our `vector_add` kernel we replace the following line:

```c
int item = threadIdx.x;
```

With this other line of code:

```c=
int item = 1;
```

What will the result of this change be?

1) Nothing changes
2) Only the first thread is working
3) Only `C[1]` is written
4) All elements of `C` are zero


### Challenge: Hidden variables

Given the following snippet of code:

```python
size = 512
vector_add_gpu((4, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))
```

What is the content of the `blockDim` and `gridDim` variables inside the CUDA vector_add kernel?

### Challenge: Scaling up

In the following code, fill in the blank to work with vectors that are larger than the largest CUDA block (i.e. 1024).

```c
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = ______________;
   C[item] = A[item] + B[item];
}
```

### Challenge: Compute prime numbers with CUDA

Given the following Python code, similar to what we have seen in the previous episode about Numba, write the missing CUDA kernel that computes all the prime numbers up to a certain upper bound.

```python
import numpy
import cupy
import math

# CPU
def all_primes_to(upper : int, prime_list : list):
    for num in range(2, upper):
        prime = True
        for i in range(2, (num // 2) + 1):
            if (num % i) == 0:
                prime = False
                break
        if prime:
            prime_list[num] = 1

upper_bound = 100_000
all_primes_cpu = numpy.zeros(upper_bound, dtype=numpy.int32)
all_primes_cpu[0] = 1
all_primes_cpu[1] = 1
%timeit all_primes_to(upper_bound, all_primes_cpu)

# GPU
check_prime_gpu_code = r'''
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
}
'''
# Allocate memory
all_primes_gpu = cupy.zeros(upper_bound, dtype=cupy.int32)

# Compile and execute code
all_primes_to_gpu = cupy.RawKernel(check_prime_gpu_code, "all_primes_to")
grid_size = (int(math.ceil(upper_bound / 1024)), 1, 1)
block_size = (1024, 1, 1)
%timeit all_primes_to_gpu(grid_size, block_size, (upper_bound, all_primes_gpu))

# Test
if numpy.allclose(all_primes_cpu, all_primes_gpu):
    print("Correct results!")
else:
    print("Wrong results!")
```

There is no need to modify anything in the code, except writing the body of the CUDA `all_primes_to` inside the `check_prime_gpu_code` string, as we did in the examples so far.

Hint: look at the body of the Python `all_primes_to` function, and map the outermost loop to the CUDA grid.


### Challenge: use shared memory to speed up the histogram

Implement a new version of the CUDA `histogram` function that uses shared memory to reduce conflicts in global memory.

```c
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;

    atomicAdd(&(output[input[item]]), 1);
}
```

Hint: for this exercise, you can safely assume that the size of output is the same as the number of threads in a block.

Hint: `atomicAdd` can be used on both global and shared memory.

## 🧠 Collaborative Notes

#### Vector addition

We have no numpy in CUDA, so we have to be much more explicit when we use it. This Python code is a bit like that:

```python=
def vector_add(A, B, C, size):
    for item in range(0, size):
        C[item] = A[item] + B[item]
    return C
```
What's inside the `for` loop above is completely independent from each other. In CUDA, the `for` loop that is written above is implicit.

How would such a thing work in CUDA?

```c=
extern "C"
__global__ void vector_add(const float *A, const float *B, float *C, const int size)
    
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
```

This function tells the compiler to recognise things from the outside.  `void` above means "do not return anything". The `*` refers to an array. We have a special threading (`threadIdx`) command that tells us where we are in the for loop. Because our problem is only 1-dimensional, the `threadIdx` only needs one of the dimensions, `x`. threadIdx is the index of which thread is executing (refers to `item`) in the Python function. 

![CUDA thread hierarchy](https://www.researchgate.net/profile/Ponnuswamy_Sadayappan/publication/220953198/figure/download/fig2/AS:393937765322784@1470933490966/Hierarchy-of-threads-in-the-CUDA-programming-model.png)

We will generate random numbers to execute this function on the gpu. The cuda code needs to be added as a string. 

```python=
import cupy

size = 1024

a_gpu = cupy.random.rand(size, dtype = cupy.float32)
b_gpu = cupy.random.rand(size, dtype = cupy.float32)
c_gpu = cupy.zeros(size, dtype = cupy.float32)

vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float *A, const float *B, float *C, const int size)
    
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
'''

vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, "vector_add")
vector_add_gpu((1, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size)) 
```
Everything is optimized for single precision on a GPU, so it's better to use `float32` than `float64`. The `r` avoids escapes.

We want everything to be contained within a single cell in the notebook. 

![Difference between CPU and GPU](https://microcontrollerslab.com/wp-content/uploads/2019/02/CPU-and-GPU.png)

We want to do vector add, on one single thread block (indicated by the first argument to `vector_add_gpu`). The second argument specifies that we will have `size` number of threads inside the thread block, only using one dimension (no need to structure the problem in three dimensions). The last argument above `(a_gpu, b_gpu, c_gpu, size)` are the arguments to the CUDA function we wrote. 

```python=
import numpy

a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu)
c_cpu = numpy.zeros(size, dtype=numpy.float32)

vector_add(a_cpu, b_cpu, c_cpu, size)

if numpy.allclose(c_cpu, cupy.asnumpy(c_gpu)):
    print("Correct results!")
```

`c_gpu` was calculated on the gpu, and on the cpu we have a version we trust. For that reason, we compare the `c_cpu` to `c_gpu`. When the code above prints `"Correct results!"`, the two are the same. 

`__global__` means `__host__` and `__device__`

When we set the size to 2048, CUDA tells use that there is an invalid argument. For most CUDA arguments, 1024 threads per block is the maximum. So how can we make CUDA do larger amounts of work?

```python=
import cupy

size = 2048

a_gpu = cupy.random.rand(size, dtype = cupy.float32)
b_gpu = cupy.random.rand(size, dtype = cupy.float32)
c_gpu = cupy.zeros(size, dtype = cupy.float32)

vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float *A, const float *B, float *C, const int size)
    
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
'''

vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, "vector_add")
vector_add_gpu((2, 1, 1), (size // 2, 1, 1), (a_gpu, b_gpu, c_gpu, size)) 
```

Is the result still correct?

```python=
import numpy

a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu)
c_cpu = numpy.zeros(size, dtype=numpy.float32)

vector_add(a_cpu, b_cpu, c_cpu, size)

if numpy.allclose(c_cpu, cupy.asnumpy(c_gpu)):
    print("Correct results!")
else:
    print("WRONG!")
```

When we execute this, we can see that only the second half of the numbers is still matching. (inspect `c_gpu` and `c_cpu` for yourself to check this) This is because the threadIdx tells us where in the Thread we are, but not where in the "work" we are. So next to the thread index, we need to keep track of the block index. 

- `threadIdx` current location within block (lowest denomination)
- `blockDim` size of the block (number of threads)
- `blockIdx` current location of block within grid (current location of block within a grid)
- `gridDim` size of the grid (number of blocks)

Together, these four numbers (which are all three-dimensional using `x`, `y` and `z` components), give you the entire geometry of your problem. Using these variables, we can change our code to make sure it works correctly. 

You always work within a single grid, and different grids do not communicate. 

Suppose we want to write a GPU code interface that's generic with any size (not just 1024 or 2048). We will always need to do a little bit of computation to make that work. For example, we may have 2049 elements. This means we do not have a multiple of the blockDim. We need to build in a bit of safety. 


```python=
import cupy

size = 2049

a_gpu = cupy.random.rand(size, dtype = cupy.float32)
b_gpu = cupy.random.rand(size, dtype = cupy.float32)
c_gpu = cupy.zeros(size, dtype = cupy.float32)

vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float *A, const float *B, float *C, const int size)
    
{
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < size) {
        C[item] = A[item] + B[item];
    }
}
'''

grid_size = (int(math.ceil(size / 1024)), 1, 1) 
block_size = (1024, 1, 1)

vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, "vector_add")
vector_add_gpu(grid_size, block_size, (a_gpu, b_gpu, c_gpu, size)) 
```

1024 refers to the maximum block size. Now we can use all kinds of crazy numbers for `size`

### Different types of GPU memory
For our understanding of the performance of the GPU we need to look at different kinds of memory. From fast to slow (roughly):

- Registers: contain some static variables to do arithmetic on. There aren't that many, so be careful with the amount of variables used per thread.
- Local memory: storage for each thread individually, used if we run out of registers. This is not very fast memory, so not needed often.
- Shared memory: shared between threads but not between blocks. We can use shared memory to have threads communicate with each other (but still not globally!). Shared memory is much faster than global memory.
- Constant memory & Texture memory: constant memory available to all threads in all thread blocks. Faster than global memory because trafic is one way.
- Global memory: Read-write for all threads. This is where we store end results of computations.

### Working with shared memory
Suppose we want to use some shared memory. We should prefix our array declaration with the `__shared__` keyword.

```c=
__shared__ float small_array[8192];
```

A little bit of useless code:

```c=
extern "C"
__global__ void vector_add(const float *A, const float *B, float *C, const int size)
{
    __shared__ float small_array[8192];
    
    int item = blockIdx.x * blockDim.x + threadIdx.x;
        
    if (item < size)
    {
        small_array[256] = item;
        C[item] = A[item] + B[item]
    }
}
```

If we want to have a variable amount of `__shared__` memory, e.g. varying with `blockDim`, we can tell CUDA that the shared memory should be allocated at run time. To this end we add the `extern` keyword:

```c=
extern "C"
__global__ void vector_add(const float *A, const float *B, float *C, const int size)
{
    extern __shared__ float small_array[];
    
    int item = blockIdx.x * blockDim.x + threadIdx.x;
        
    if (item < size)
    {
        small_array[256] = item;
        C[item] = A[item] + B[item]
    }
}
```

Now we need to tell CUDA how much memory to use:

```python=
size = 2048
float_size = cupy.dtype(cupy.float32).itemsize
vector_add_gpu(
    (2, 1, 1), (size // 2, 1, 1),
    (a_gpu, b_gpu, c_gpu, size),
    shared_mem=((size // 2) * float_size))
```

Note: we use `itemsize` member of the `cupy.dtype` class to find the size of `float32`, which should be 4. The `shared_mem` parameter accepts a value in bytes.

### Histogram
Let's look at a more functional example of using shared memory: computing a histogram. First we implement in Python a naive version, both for prototyping in an easier environment and checking our result later on.

```python=
def histogram(input_array, output_array):
    for item in input_array:
        output_array[item] = output_array[item] + 1
    return output_array
```

We generate some random numbers between 0-255. The output array should then have 256 entries so that we can count all occurances.

```python=
import numpy

input_array = numpy.random.randint(256, size=2048, dtype=numpy.int32)
output_array = numpy.zeros(256, dtype=numpy.int32)
```

To CUDA

```c=
__global__ void histogram(const int * input, int * output)
{
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    output[input[item]] = output[input[item]] + 1;
}
```

This won't work, because all threads will try to update their element at the same time! We need to use atomics to fix the situation.

```c=
__global__ void histogram(const int * input, int * output)
{
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&(output[input[item]]), 1);
}
```

This will be very slow. Now using shared memory (asuming histogram output size of 256):

```c=
__global__ void histogram(const int * input, int * output)
{
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ int output_shared[256];
    if (threadIdx.x < 256)
    {
        output_shared[threadIdx.x] = 0;
    }
    __syncthreads();
    
    atomicAdd(&(output_shared[input[item]]), 1);
    __syncthreads();
    
    if (threadIdx.x < 256)
    {
        atomicAdd(&(output[threadIdx.x]), output_shared[threadIxd.x]);
    }
}
```

Complete running code for the histogram example.

```python=
import math
import numpy
import cupy

# input size
size = 2048

# allocate memory on CPU and GPU
input_gpu = cupy.random.randint(256, size=size, dtype=cupy.int32)
input_cpu = cupy.asnumpy(input_gpu)
output_gpu = cupy.zeros(256, dtype=cupy.int32)
output_cpu = cupy.asnumpy(output_gpu)

# CUDA code
histogram_cuda_code = r'''
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    extern __shared__ int temp_histogram[];
 
    // Initialize shared memory and synchronize
    temp_histogram[threadIdx.x] = 0;
    __syncthreads();

    // Compute shared memory histogram and synchronize
    atomicAdd(&(temp_histogram[input[item]]), 1);
    __syncthreads();

    // Update global histogram
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
'''

# compile and setup CUDA code
histogram_gpu = cupy.RawKernel(histogram_cuda_code, "histogram")
threads_per_block = 256
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

# execute code on CPU and GPU
histogram_gpu(grid_size, block_size, (input_gpu, output_gpu), shared_mem=(threads_per_block * cupy.dtype(cupy.int32).itemsize))
histogram(input_cpu, output_cpu)

# compare results
numpy.allclose(output_cpu, output_gpu)
```

## :question: Open Questions

* Q: I lack motivation to learn CUDA, because I do not have a device on which I can run it afterwards. I'd rather learn OpenCL. But I see that CUDA has better support for Python, whereas OpenCL forces you to go low-level. But today we are going low-level with CUDA. Why not with OpenCL?
    * A: The reason is that CUDA is widely used in practice, while OpenCL is barely used (except from CS researchers, I did my whole PhD thesis in OpenCL) at all. Unfortunately no one ever backed-up OpenCL for real. Apple created it and then abandoned it, Intel wanted to support it but decided it is too low level and went with SyCL and OneAPI, NVIDIA supports it but barely, AMD used to support in more in the past but moved to HIP.
    * Anyway everything we are teaching about CUDA works in OpenCL just changing keywords.
    * So if OpenCL is a dead end, how do yor program non-NVIDIA GPUs then?
        * The reality is that most GPUs are NVIDIA. AMD uses HIP that is CUDA with the word CUDA replaced by the word HIP in all function calls. Intel is still not clear on what to support and as I said come up with OneAPI from within C++. Apple has Metal but is very niche and more focused on graphics.
    * Very helpful, thanks.
        * In my TODO list there is still to develop a parallel course to this one with OpenCL, but I lack the time
    * I can imagine that there is less motivation to teach OpenCL in light of what you said! 
* Q: Why does Cuda provide both the blocks and threads abstraction? You now have to deal with both block and thread indices, having to take safety measures and having to deal with a maximum number of threads per block. Why not only threads?
    * JH: The hardware would become more complicated. This abstraction is the result of a compromise between usability and performance. Threads within the same block always execute the same instructions, while different thread blocks may be doing different work. The other advantage of this design is *shared memory*. Because shared memory is only shared within the same thread block, it can be much faster than global memory.
* Q: Can we say that speed-wise: `register > local memory > shared memory > global memory`?
    * JH: Local memory is the odd one out, since it is slower than shared memory.
* Q: Can we say that capacity-wise: `global memory > shared memory > local memory > register`?
    * JH: Local memory lives at the same place as global memory, except that it is reserved for local use.

## Feedback
[Post-workshop survey](https://www.surveymonkey.com/r/78GNSJB)

### What went well :+1:
 
### What could be improved :-1:

## Videos about GPUs

[![Graphics Processing Unit (GPU)](http://img.youtube.com/vi/bZdxcHEM-uc/0.jpg)](https://www.youtube.com/watch?v=bZdxcHEM-uc "Graphics Processing Unit (GPU)")

[![GPU Hardware Introduction](http://img.youtube.com/vi/FcS_kQOIykU/0.jpg)](https://www.youtube.com/watch?v=FcS_kQOIykU "GPU Hardware Introduction")


## 📚 Resources

* [Upcoming eScience Center workshops](https://www.esciencecenter.nl/digital-skills/)
* [HackMD Markdown Guide](https://www.markdownguide.org/tools/hackmd/)
* [JupyterHub Guide](https://servicedesk.surfsara.nl/wiki/display/WIKI/User+Manual+-+Student)
* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* [Post-workshop Survey](https://www.surveymonkey.com/r/78GNSJB)
