# CFD-Sod-Shock-Tube

## Getting Started

### Introduction
This code mainly solves sod shock tube using finite volume method.

Three spatial reconstruction methods are implemented:
- TVD-GVC
- 2 order TVD
- 5 order WENO

The following Riemann solvers are provided:
- Steger-Warming
- HLL
- Lax-Friedrichs
- Exact answers solved by open source package [lanl/ExactPack](https://github.com/chairmanmao256/Python-shock-tube)

### Prerequisites
- Python
- Numpy
- Pandas
- [ExactPack](https://github.com/chairmanmao256/Python-shock-tube)

## Usage
 ```bash
python test.py
 ```
Through this file, we can get all the results for combinations of reconstruction methods and Riemann solvers at time t=0.14 and with different cell numbers and computational fields.


## Reference
- https://github.com/chairmanmao256/Python-shock-tube
  Parts of the code (IO, code structure and TVD reconstruction) are adapted from [chairmanmao256/Python-shock-tube](https://github.com/chairmanmao256/Python-shock-tube), with modifications.
