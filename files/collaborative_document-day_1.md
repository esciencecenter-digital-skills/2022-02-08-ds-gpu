![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document. Day 1, February 8

2022-02-08-ds-gpu

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](https://hackmd.io/@isazi/rk-rL6vRY)

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

## 👩‍🏫👩‍💻🎓 Instructors

Alessio Sclocco, Hanno Spreeuw, Johan Hidding

## 🧑‍🙋 Helpers

Lieke de Boer

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call

Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city

## 🗓️ Agenda

| time | what |
|---|---|
| 09:00 |	Welcome and icebreaker |
| 09:15 |	Introduction |
| 09:30 |	Convolve an image with a kernel on a GPU using CuPy |
| 10:15 |	Coffee break |
| 10:30 |	Running CPU/GPU agnostic code using CuPy |
| 11:30 |	Coffee break |
| 11:45 |	Run your Python code on a GPU using Numba |
| 12:45 |	Wrap-up |
| 13:00 |	END |

## 🔧 Exercises

### Challenge: fairer runtime comparison CPU vs. GPU

Compute again the speedup achieved using the GPU, but try to take also into account the time spent transferring the data to the GPU and back.

Hint: to copy a CuPy array back to the host (CPU), use the `cp.asnumpy()` function.


#### Some solution
```python=
def transfer_compute_transferback():
    image_gpu = cp.asarray(image)
    gauss_gpu = cp.asarray(gauss)
    convolved_image_using_GPU = convolve2d_gpu(image_gpu, gauss_gpu)
    convolved_image_using_GPU_copied_to_host = cp.asnumpy(convolved_image_using_GPU)
   
%timeit transfer_compute_transferback()
```

Copying the data can in this case take about a large factor of the time to compute the convolution!

### Challenge: compute prime numbers

Write a new function `find_all_primes_cpu_and_gpu` that uses `check_prime_gpu_kernel` instead of the inner loop of `find_all_primes_cpu`. How long does this new function take to find all primes up to 10000?
How long does it take to find all primes up to 10000?

#### Some answer

```python=
def find_all_primes_cpu_and_gpu(upper):
    all_prime_numbers = []
    for num in range(2, upper):
        result = np.zeros((1), np.int32)
        check_prime_gpu_kernel[1,1](num, result)
        if result[0] > 0:
            all_prime_numbers.append(num)
    return all_prime_numbers
   
%timeit find_all_primes_cpu_and_gpu(10000)
```

This may take about five seconds to run! The truth is that the GPU is slower than the CPU. Only by running thousands of threads at the same time can it be faster than the CPU.

## 🧠 Collaborative Notes

### Introduction
- Graphics Processing Unit: computing the color of all pixels on a screen at the same time. The GPU is a dedicated processor to accellerate the computation of video images. It computes every pixel for your 4k monitor **at the same time**, 60 times per second, i.e. in parallel.

Let's try to sort a list of numbers using the CPU.

```python=
import numpy as np
size = 4096*4096
input = np.random.random(size).astype(np.float32)

%timeit output = np.sort(input)
```

Takes about 1-2 seconds. Now with the GPU,

```python=
import cupy as cp
input_gpu = cp.asarray(input)

%timeit output_gpu = cp.sort(input_gpu)
```

Takes about 1-2 **mili**seconds, so we're on the order of 1000 times faster using a GPU in this case.

Later today you can watch this video to learn more about GPUs.

[![Graphics Processing Unit (GPU)](http://img.youtube.com/vi/bZdxcHEM-uc/0.jpg)](https://www.youtube.com/watch?v=bZdxcHEM-uc "Graphics Processing Unit (GPU)")

### Convolutions
- A **convolution** is a computational / mathematical operation that takes a weighted average of a region over some image. In one dimension an example would be a running average. In two dimensions, you might think of smoothing an image. The concept generalises to any dimension.

![Example of convolution](https://carpentries-incubator.github.io/lesson-gpu-programming/fig/2D_Convolution_Animation.gif)

For each pixel in the output image, we need to take a little square area of the input into account. If the convolution kernel contains K pixels and the input image N pixels, the total computation takes `N*K` steps.

#### On the CPU

```python=
tile = np.zeros((16, 16))
tile[8,8] = 1
image = np.tile(tile, (128, 128))

# or alternatively:
# image = np.zeros((2048, 2048), dtype=np.float32)
# image[8::16, 8::16] = 1
```

```python=
import pylab as pyl
import matplotlib
%matplotlib inline

pyl.imshow(image[0:64,0:64])
pyl.show()
```

Now we create the convolution kernel:

```python=
x, y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
dist2 = x**2 + y**2
gauss = np.exp(-dist)

pyl.imshow(gauss)
```

We can compute the convolution using SciPy:

```python=
from scipy.signal import convolve2d
output_image = convolve2d(image, gauss)   
# this adds a boundary. to keep same domain: mode='same'
%timeit convolve2d(image, gauss)
```

And look at the result:

```python=
pyl.imshow(output_image[:64, :64])
```

#### On the GPU
There is the `cupyx` library that contains lots of algorithms from numpy/scipy.

CPU and GPU are twp separate entities, with different memories and a connection between them.

![CPU and GPU are two separate entities, each with its own memory.](https://carpentries-incubator.github.io/lesson-gpu-programming/fig/CPU_and_GPU_separated.png)

- **We need to copy our data to the GPU.**

```python=
from cupyx.scipy.signal import convolve2d as convolve2d_gpu

image_gpu = cp.asarray(image)
gauss_gpu = cp.asarray(gauss)

output_image_gpu = convolve2d_gpu(image_gpu, gauss_gpu)
```

It is common practice to have a suffix on your variables to tell if they're on the host or device (gpu) `_h` and `_d` are common. Here we use `_gpu` to denote the GPU mirrors of host data.

To visualize the result we need to copy back from the GPU to the host.

```python=
output_image_host = cp.asnumpy(output_image_gpu)
pyl.imshow(output_image_host[:64,:64])
```

Time the computation:

```python=
%timeit -n 100 -r 10 convolve2d_gpu(image_gpu, gauss_gpu)
```

#### Check your output

```python=
np.allclose(output_image_gpu.get(), output_image)
```

Should give `True`.

### Example of memory mistake? 1-d convolution

```python=
image_1d = np.ravel(image)
gauss_1d = np.ravel(gauss)
np.convolve(image_1d, gauss_1d)
```

```python=
image_1d_gpu = cp.asarray(image_1d)
gauss_1d_gpu = cp.asarray(gauss_1d)
np.convolve(image_1d_gpu, gauss_1d_gpu)
```

Still works... so NumPy is being flexible. Try to be consistent if you want to understand the performance of your code.

### Using Numba with a GPU
We'll be computing prime numbers.

```python=
def find_primes(upper):
    all_primes = []
    for i in range(2, upper):
        for j in range(2, i//2 + 1):
            if i%j == 0:
                break
        else:
            all_primes.append(i)
    return all_primes
```

```python=
%timeit find_primes(10000)
```

Let's make this faster using Numba. Numba employs a strategy called Just-In-Time compilation. The function will be compiled to machine code the first time it is called. This will make the first call a bit slower, but subsequent calls will be extremely fast!

```python=
import numba

find_primes_jit = numba.jit(find_primes)

# trigger compile
find_primes_jit(0)

%timeit find_primes_jit(10000)
```

Now to the GPU!

```python=
@numba.cuda.jit
def check_prime_gpu_kernel(number, result):
    result[0] = 0
    for j in range(2, number//2 + 1):
        if number%j == 0:
            break
    else:
        result[0]=number
```

```python=
result = np.zeros((1))

check_prime_gpu_kernel[1,1](11, result)
```

This is using the GPU to check if a single number is prime. This goes against the grain of what a GPU is actually good at: running thousands of threads in parallel.

First, lets see how `numba` can help us **vectorize** a small function.

```python=
from numba import vectorize
import math

@vectorize
def log_of_sum(x, y):
    return math.log(x + y)

log_of_sum(6, 7)
# 2.564....
log_of_sum(np.random.random(10), np.random.random(10))
# array([ ... ])
```

We now have a function that is generic over numbers and arrays! The execution model of such a function (Numpy UFunc) is similar to what we have when working on a GPU.

```python=
@vectorize([numba.int32(numba.int32)], target="cuda")
def check_prime(number):
    for j in range(2, number // 2 + 1):
        if number % j == 0:
            return 0
    else:
        return number
```

```python=
%timeit -n 1 -r 1 check_prime(np.arange(10**6, dtype=np.int32))
```

To reduce to a list of primes:

```python=
nums = check_prime(np.arange(10**6, dtype=np.int32))
nums = nums[nums != 0]
```

## Videos about GPUs (watch before day 2)

[![Graphics Processing Unit (GPU)](http://img.youtube.com/vi/bZdxcHEM-uc/0.jpg)](https://www.youtube.com/watch?v=bZdxcHEM-uc "Graphics Processing Unit (GPU)")

[![GPU Hardware Introduction](http://img.youtube.com/vi/FcS_kQOIykU/0.jpg)](https://www.youtube.com/watch?v=FcS_kQOIykU "GPU Hardware Introduction")

## :question: Open Questions

* Q: How do you install cuda on macos (intel) and macos (M1)?
    * A: I have never done that, since Apple is not using NVIDIA GPUs in their own computers for a while now. To run code on GPUs you will need to either use a different language than CUDA (CUDA is basically NVIDIA only), and one example of this other language is OpenCL, or develop your code on Mac but run it on a different machine with Linux/Windows.
    * This is the answer from NVIDIA: *NVIDIA® CUDA Toolkit 11.6 no longer supports development or running applications on macOS. While there are no tools which use macOS as a target environment, NVIDIA is making macOS host versions of these tools that you can launch profiling and debugging sessions on supported target platforms.*
    * (Dirk): I am not sure what NVIDIA says here. I can do something on macos, but what exactly?
        * A: install something that lets you run code remotely on a different machine for debug and profiling.
    * Still, the M1 system on a chip seems excellent hardware to do GPU computing. Is there a practical way to use it?
        * A: yes but not with CUDA. The M1 chip has a good GPU. In this course we use CUDA although tomorrow we will mention other languages that can be used (and that can also work on the M1 GPU).

* Q: is it faster to slice a cupy array and then transfer it from GPU memory, or the the other way around?
    * A: it depends on the CuPy implementations
    * A: based on my initial trials it is extremely faster to slice it first and then transfer it:
    * ![](https://i.imgur.com/NsVJE2Z.png)
    * A: so it seems that it is not straight forward :). based on [the cupy official website](https://cupy.dev/) slicing is much faster compared to NumPy. However, based on [this](https://stackoverflow.com/questions/60956806/slicing-a-300mb-cupy-array-is-5x-slower-than-numpy) if the process is a one time slice, then it is faster to do it on CuPy, if it is a batch generation process that needs to slice all the data, then NumPy is faster since we don't save any computation time on the transfer. Also, it extremely depends on the size of array as well, since if the array can fill up the memory channels, then it will definitly be faster on CuPy.

* Q: CuPy / NumPy interoperability
    * A: https://docs.cupy.dev/en/v8.2.0/reference/interoperability.html#numpy

* Q: (Dirk) Can dictionary lookups be sped up? I've read that Python dicts are already very efficient (same implementation as PyPy)
    * A: Indeed they are. Dictionary look-ups are at the core of Python's object system, so they have to be fast. The question if the look-up can be sped up is a bit of a curve-ball. It is not something that you would have a GPU typically do, but if you need it inside a kernel it is there in most frameworks.
    * My use case is: I have dicts with numbers as keys usually between 1 and 10,000,000. I could turn them into arrays, but some of these dicts are really sparse. The values are numbers or strings (aargh)
    * What are the values? You could have two arrays: one with the keys and one with the values. Having strings in CUDA is not very nice. If you can guarantee that they are short you could try storing those in fixed-length containers.
    * Interesting! I tried both things, also the two arrays. But then the python arithmetic is getting too slow, so I would have to Cythonize that.
    * Are you sure your problem is compute-bound? I.e. is it slow because you need to do lots of computing, or is it just a lot of data?
    * I profiled it. All time I gained by using efficient data structures I lost by using plain python to compute indices and to call functions.
    * Then moving away from Python might be a necessary step. Julia could be a nice alternative. (Hard to say without more info)
    * I tried Julia, did not help in this case. But maybe Rust  
    * If Julia doesn't help, it is hard to imagine how Rust would.
    * Ah, when I tried Julia, I was still using dicts. Not the array solution.
    * Yes, choosing the right data structure can make or break any algorithm.

* Q: What is the difference between @cuda.jit and @vectorize(target="cuda")?
    * @vectorize will use @cuda.jit under the hood, @vectorize gives us a nicer interface (i.e. we have to think less).

## Feedback

### What went well :+1:

### What could be improved :-1:

## 📚 Resources

* [Upcoming eScience Center workshops](https://www.esciencecenter.nl/digital-skills/)
* [HackMD Markdown Guide](https://www.markdownguide.org/tools/hackmd/)
* [JupyterHub Guide](https://servicedesk.surfsara.nl/wiki/display/WIKI/User+Manual+-+Student)
* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [Numba User Guide](https://numba.pydata.org/numba-doc/latest/user/index.html)
* [Fellowship Programme](https://www.esciencecenter.nl/fellowship-programme/)
* [eScience Center website](https://www.esciencecenter.nl/)
