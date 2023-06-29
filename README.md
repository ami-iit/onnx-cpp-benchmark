# onnx-cpp-benchmark

Simple tool to profile onnx inference with C++ APIs.

## Installation

### With conda-forge dependencies

~~~
mamba create -n onnxcppbenchmark compilers cli11 onnxruntime cmake ninja pkg-config
mamba activate onnxcppbenchmark
git clone https://github.com/ami-iit/onnx-cpp-benchmark
cd onnx-cpp-benchmark
mkdir build
cd build
cmake -GNinja ..
~~~

And then add `onnx-cpp-benchmark/build` to the PATH env variable.

## Usage

Download a simple `.onnx` file and run the benchmark on it.

```shell
curl -L https://huggingface.co/ami-iit/mann/resolve/3a6fa8fe38d39deae540e4aca06063e9f2b53380/ergocubSN000_26j_49e.onnx -o ergocubSN000_26j_49e.onnx
# Use default options
onnx-cpp-benchmark ergocubSN000_26j_49e.onnx

# Specify custom options
onnx-cpp-benchmark ergocubSN000_26j_49e.onnx --iterations 100 --batch_size 5 --backend onnxruntimecpu
```

Current supported backends:
* `onnxruntimecpu`  : [ONNX Runtime](https://onnxruntime.ai/) with CPU
* `onnxruntimecuda`  : [ONNX Runtime](https://onnxruntime.ai/) with CUDA


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[BSD 3-Clause](https://choosealicense.com/licenses/bsd-3-clause/)
