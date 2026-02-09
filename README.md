<h1 align="center">
<br>
<strong>Mneme.jl: Parallel Out-of-Core Tabular Data Preprocessing in Julia</strong>
</h1>


[![Julia](https://img.shields.io/badge/Julia-1.12-purple?logo=julia&logoColor=white)](https://julialang.org) ![HPC](https://img.shields.io/badge/High--Performance%20Computing-HPC-green)

**Mneme.jl** is a high-level parallel preprocessing framework for large-scale tabular data. Built on top of torcjulia, Mneme.jl employs an MPI-based hybrid parallelism scheme inspired by the MapReduce paradigm, which combines multiprocessing and multithreading to efficiently support out-of-core preprocessing of tabular datasets on multi-node systems (clusters).

Mneme.jl can be considered as the Julia implementation of the [Mneme](https://github.com/CEID-HPCLAB/Mneme) Python library. The project is under active development, and some features of the original Python library are not yet supported.  
