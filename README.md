# Comparison of Lines of Code using [KaMPIng][kamping], Plain MPI, and [Thrill][thrill]

This repository contains the code for two suffix array construction algorithms: [Prefix Doubling](https://doi.org/10.1137/0222058) (PD) and [DCX](https://dl.acm.org/doi/10.1145/1217856.1217858).
These algorithms have been implemented using [KaMPIng][kamping] (PD and DCX), Plain MPI (PD and DCX), and [Thrill][thrill] (PD).

The lines of code (LOC) necessary to implement the two algorithms using the different distributed memory programming frameworks are listed below.
We measured the lines of code using [`cloc`](https://github.com/AlDanial/cloc) 2.00.
All code files have been formatted using [`clang-format`](https://releases.llvm.org/14.0.0/tools/clang/docs/ClangFormat.html) 14 using the default Google style (`clang-format --style=Google`).

|    | [KaMPIng][kamping] | plain MPI | [Thrill][thrill] |
|----|--------------------|-----------|------------------|
| PD | 163                |426 (+1442)| 266              |
| DCX| 1264               |1396       | ---              |

## Prefix Doubling
Note that the prefix doubling implementations are copied from three different projects.
To keep the overhead of this repository small, we removed all code not directly part of the algorithm, e.g., `main` functions and benchmark utility.
Therefore, the code included here is not expected to be executed.
If you want to execute the algorithms, we refer to the corresponding repositories ([KaMPIng implementation](https://github.com/kamping-site/kamping/blob/main/examples/applications/suffix-sorting/prefix_doubling.hpp), [plain MPI implementation](https://github.com/kurpicz/dsss/blob/master/dsss/suffix_sorting/prefix_doubling.hpp), and [Thrill implementation](https://github.com/thrill/thrill/blob/master/examples/suffix_sorting/prefix_doubling.cpp)).

To reproduce the measured lines of code, you can use the following commands:

```bash
cd prefix_doubling_comparison/
echo -e "\e[32mLines of Code: PD KaMPIng Implementation\e[0m"
cloc kamping_prefix_doubling.hpp
echo -e "\e[32mLines of Code: PD Plain MPI Implementation\e[0m"
cloc mpi_prefix_doubling.hpp
echo -e "\e[32mLines of Code: Plain MPI Implementation MPI Wrapper\e[0m"
cloc dsss_mpi
echo -e "\e[32mLines of Code: PD Thrill Implementation\e[0m"
cloc thrill_prefix_doubling.cpp
cd ..
```

## DCX
Implementation of the [DCX](https://dl.acm.org/doi/10.1145/1217856.1217858) suffix array construction algorithm.
The original [implementation](src/mpi_dc.cpp) is by Timo Bingmann and consists of 1396 lines of code, when removing code shared with the [KaMPIng][kamping] implementation.
In our [KaMPIng][kamping] [implementation](src/kamping_dc.cpp), we replaced all plain MPI calls with our wrapper.
This mainly resulted in less boilerplate code and 1264 lines of code, i.e., 9.5% less code.

To reproduce the measured lines of code, you can use the following commands:

```bash
cd src/
echo -e "\e[32mLines of Code: DCX KaMPIng Implementation\e[0m"
cloc kamping_dc.cpp
echo -e "\e[32mLines of Code: Plain MPI Implementation\e[0m"
cloc mpi_dc.cpp
cd ..
```

Since this repository is based on these two implementations, they can be build and executed.

```bash
cmake -B build
cmake --build build
mpirun -np <# MPI threads> ./build/src/[kampingDCX|pDCX] [3/7/13] <input_file>
```

[kamping]: https://github.com/kamping-site/kamping "KaMPIng Repository"
[thrill]: https://project-thrill.org "Thrill's website"