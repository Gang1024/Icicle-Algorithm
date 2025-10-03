# Icicle-Algorithm
A Python implementation of the Icicle Algorithm (IA), a newly proposed physics-based metaheuristic optimization algorithm.

This repository provides the official Python implementation of the **Icicle Algorithm (IA)**, a newly proposed physics-based metaheuristic optimization algorithm.  
The algorithm is inspired by the natural formation process of icicles, which includes four stages: **melting, movement, solidification, and falling**.  
By simulating this mechanism, IA achieves a good balance between exploration and exploitation, effectively avoiding premature convergence and improving convergence speed.  

The implementation is based on the [`mealpy`](https://github.com/thieu1995/mealpy) library, which provides a flexible framework for metaheuristic algorithms.

### Features
- Pure Python implementation
- Built on top of `mealpy` library
- Easy to extend and modify
- Includes benchmark experiments from CEC-2017 and CEC-2022 test suites
- Applied to engineering design problems

### Requirements
- Python >= 3.11
- `mealpy` >= 3.0.1
- `numpy`, `matplotlib`, `opfunu`

### Usage Example
```python
import numpy as np
from mealpy import FloatVar
from IA import OriginalIA

def objective_function(solution):
    return np.sum(solution**2)

problem_dict = {
    "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30),
    "minmax": "min",
    "obj_func": objective_function,
}

model = OriginalIA(epoch=1000, pop_size=50)
g_best = model.solve(problem_dict, "thread", n_workers=32)
print(f"Best solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
