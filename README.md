# IMP-UQFE

## Scheme

The following project implements the unbounded Quadratic Functional Encryption Scheme as described on p.52 of the following paper

* [Unbounded Predicate Inner Product Functional Encryption from Pairings](https://eprint.iacr.org/2023/483.pdf)

## Disclaimer:
Disclaimer
The code provided is for educational and research purposes only. It has not been reviewed, tested, or verified for security, reliability, or suitability in any environment. The code MUST NOT be used in production systems or for any purpose where failure, errors, or vulnerabilities could cause harm or damage.

The authors and contributors make no warranties or guarantees, express or implied, regarding the code, including but not limited to its correctness, safety, or fitness for any particular purpose. All use of this code is at your own risk. The authors explicitly disclaim any and all liability for direct, indirect, incidental, or consequential damages arising from the use, misuse, or inability to use the code.

## Structure

* [uqfe.py](./uqfe.py) contains the implementation of the scheme
* [qfehelpers.py](./qfehelpers.py) helper functions used by the scheme
* [benchmark.py](./benchmark.py) calls the scheme in different ways and provides benchmarks

## Prerequisites

* Docker or Podman (if using podman replace the "docker" with "podman" in the instructions below)

## Usage

Clone the repository

```shell
git clone https://github.com/karimib/imp-uqfe-demo-charm.git
cd imp-uqfe-demo-charm
```

Build the image (assuming your in the root directory)

```shell
docker build -t uqfedemo:v1 .
```

Create a container from the image

```shell
docker run uqfedemo:v1 
```

Mount a volume to save benchmark csv

````shell
docker run -v "${PWD}/results:/data" uqfedemo:v1 
````

## Benchmarking

The [benchmark.py](./benchmark.py) file contains two methods for benchmarking

| Method | Description |
| --- | --- |
| simulation_fixed_vectors() | Simulation with small vector values that are between 1 and 3 and increasing value k and increasing vector lengths m and n |
| simulation_fixed_k() | Simulation with a fixed k value and fixed vector lengths but with vector values between 1 and p |

to run them you have to uncomment line 244 and line 245 and build a new image as follows:

Build the image

```shell
docker build -t uqfedemo:v1 .
```

Then mount a volume to save the benchmark csv to your disk: 

````shell
docker run -v "${PWD}/results:/data" uqfedemo:v1 
````
