# pDCX
Implementation of the [DCX](https://dl.acm.org/doi/10.1145/1217856.1217858) suffix array construction algorithm.
The original [implementation](src/mpi_dc.cpp) is by Timo Bingmann and consists of 2055 lines of code, when removing code shared with the [KaMPIng][kamping] implementation.
In our [KaMPIng][kamping] [implementation](src/kamping_dc.cpp), we replaced all plain MPI calls with our wrapper.
This mainly resulted in less boilerplate code and 1832 lines of code, i.e., 10.85% less code.

[kamping]: https://github.com/kamping-site/kamping "KaMPIng Repository"
