/**
 *
 * pDCX
 *
 * MPI-distributed and parallel suffix sorter using difference cover.
 *
 * Written by Timo Bingmann in 2012 loosely based on the previous work
 * by Fabian Kulla in 2006.
 * Changed to using KaMPIng MPI-bindings by Florian Kurpicz in 2024.
 *
 */

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <mpi.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

//#include <proc/readproc.h>

#include "sachecker.h"
#include "yuta-sais-lite.h"
#include "common.h"

// KaMPIng includes
#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/barrier.hpp>
#include <kamping/collectives/bcast.hpp>
#include <kamping/collectives/gather.hpp>
#include <kamping/collectives/scan.hpp>
#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>
#include <kamping/p2p/recv.hpp>
#include <kamping/p2p/send.hpp>

template <typename DCParam, typename alphabet_type>
class pDCX {
 public:
  typedef unsigned int uint;

  // **********************************************************************
  // * global parameters

  static const unsigned int X = DCParam::X;
  static const unsigned int D = DCParam::D;

  // static const unsigned int DCH[X] = { 3, 6, 6, 5, 6, 5, 4 };	// additional
  // chars in tuple static const unsigned int DCD[X] = { 0, 0, 1, 0, 3, 2, 1 };
  // // depth to sort chars before using first rank

  // static const unsigned int inDC[X] = { 1, 1, 0, 1, 0, 0, 0 };

  static const bool debug = false;
  static const bool debug_input = false;
  static const bool debug_rebalance = false;
  static const bool debug_sortsample = false;
  static const bool debug_nameing = false;
  static const bool debug_recursion = false;
  static const bool debug_rec_selfcheck = false;
  static const bool debug_finalsort = false;

  static const bool debug_compare = false;

  static const bool debug_checker1 = false;
  static const bool debug_checker2 = false;

  static const bool debug_output = false;

  // **********************************************************************
  // * tuple types

  class Pair {
   public:
    uint index;
    uint name;
    unsigned char unique;

    bool operator<(const Pair& a) const { return (index < a.index); }

    static inline bool cmpName(const Pair& a, const Pair& b) {
      return (a.name < b.name);
    }

    static inline bool cmpIndexModDiv(const Pair& a, const Pair& b) {
      return (a.index % X < b.index % X) ||
             ((a.index % X == b.index % X) && (a.index / X < b.index / X));
    }

    friend std::ostream& operator<<(std::ostream& os, const Pair& p) {
      return (os << "(" << p.index << "," << p.name << "," << int(p.unique)
                 << ")");
    }
  };

  class Triple {
   public:
    uint rank1;
    uint rank2;
    alphabet_type char1;

    bool operator<(const Triple& a) const { return (rank1 < a.rank1); }

    friend std::ostream& operator<<(std::ostream& os, const Triple& p) {
      return (os << "(" << p.rank1 << "," << p.rank2 << "," << strC(p.char1)
                 << ")");
    }
  };

  class TupleS {
   public:
    alphabet_type chars[X];
    uint index;

    bool operator<(const TupleS& o) const {
      for (unsigned int i = 0; i < X; ++i) {
        if (chars[i] == o.chars[i]) continue;
        return chars[i] < o.chars[i];
      }
      return (index < o.index);
    }

    static inline bool cmpIndex(const TupleS& a, const TupleS& b) {
      return (a.index < b.index);
    }

    bool operator==(const TupleS& o) const {
      for (unsigned int i = 0; i < X; ++i) {
        if (chars[i] != o.chars[i]) return false;
      }
      return true;
    }

    friend std::ostream& operator<<(std::ostream& os, const TupleS& t) {
      os << "([";
      for (unsigned int i = 0; i < X; ++i) {
        if (i != 0) os << " ";
        os << strC(t.chars[i]);
      }
      os << "]," << t.index << ")";
      return os;
    }
  };

  struct TupleN {
    alphabet_type chars[X - 1];
    uint ranks[D];
    uint index;

    bool operator<(const TupleN& a) const {
      return cmpTupleNdepth<X - 1>(*this, a);
    }

    bool operator==(const TupleN& o) const {
      for (unsigned int i = 0; i < X - 1; ++i) {
        if (chars[i] != o.chars[i]) return false;
      }
      for (unsigned int i = 0; i < D; ++i) {
        if (ranks[i] != o.ranks[i]) return false;
      }
      if (index != o.index) return false;
      return true;
    }

    bool operator!=(const TupleN& o) const { return !(*this == o); }

    friend std::ostream& operator<<(std::ostream& os, const TupleN& t) {
      os << "(c[";
      for (unsigned int i = 0; i < X - 1; ++i) {
        if (i != 0) os << " ";
        os << strC(t.chars[i]);
      }
      os << "],r[";
      for (unsigned int i = 0; i < D; ++i) {
        if (i != 0) os << " ";
        os << t.ranks[i];
      }
      os << "]," << t.index << ")";
      return os;
    }
  };

  template <int Depth>
  static inline bool cmpTupleNdepth(const TupleN& a, const TupleN& b) {
    for (unsigned int d = 0; d < Depth; ++d) {
      if (a.chars[d] == b.chars[d]) continue;
      return (a.chars[d] < b.chars[d]);
    }

    // ranks must always differ, however for some reason a == b is possible.
    assert(a.ranks[0] != b.ranks[0] || a.index == b.index);

    return (a.ranks[0] < b.ranks[0]);
  }

  static inline bool cmpTupleNranks(const TupleN& a, const TupleN& b) {
    // ranks must always differ, however for some reason a == b is possible.
    assert(a.ranks[0] != b.ranks[0] || a.index == b.index);

    return (a.ranks[0] < b.ranks[0]);
  }

  template <int MaxDepth, typename Tuple>
  static inline void radixsort_CI(Tuple* array, uint n, size_t depth,
                                  size_t K) {
    if (n < 32) {
      std::sort(array, array + n);
      return;
    }

    if (depth == MaxDepth) {
      // still have to finish sort of first rank as tie breaker
      std::sort(array, array + n, TupleS::cmpIndex);
      return;
    }

    size_t bucketsize[K];
    memset(bucketsize, 0, K * sizeof(size_t));
    alphabet_type* oracle = (alphabet_type*)malloc(n * sizeof(alphabet_type));
    for (size_t i = 0; i < n; ++i) oracle[i] = array[i].chars[depth];
    for (size_t i = 0; i < n; ++i) {
      assert(oracle[i] < K);
      ++bucketsize[oracle[i]];
    }
    ssize_t bucketindex[K];
    bucketindex[0] = bucketsize[0];
    size_t last_bucket_size = bucketsize[0];
    for (unsigned i = 1; i < K; ++i) {
      bucketindex[i] = bucketindex[i - 1] + bucketsize[i];
      if (bucketsize[i]) last_bucket_size = bucketsize[i];
    }
    for (size_t i = 0, j; i < n - last_bucket_size;) {
      while ((j = --bucketindex[oracle[i]]) > i) {
        std::swap(array[i], array[j]);
        std::swap(oracle[i], oracle[j]);
      }
      i += bucketsize[oracle[i]];
    }
    free(oracle);

    size_t bsum = 0;
    for (size_t i = 0; i < K; bsum += bucketsize[i++]) {
      if (bucketsize[i] <= 1) continue;
      radixsort_CI<MaxDepth>(array + bsum, bucketsize[i], depth + 1, K);
    }
  }

  template <int MaxDepth, typename Tuple>
  static inline void radixsort_CI2(Tuple* array, uint n, size_t depth,
                                   size_t K) {
    if (n < 32) {
      std::sort(array, array + n);
      return;
    }

    if (depth == MaxDepth) {
      // still have to finish sort of first rank as tie breaker
      std::sort(array, array + n, cmpTupleNranks);
      return;
    }

    size_t bucketsize[K];
    memset(bucketsize, 0, K * sizeof(size_t));
    alphabet_type* oracle = (alphabet_type*)malloc(n * sizeof(alphabet_type));
    for (size_t i = 0; i < n; ++i) oracle[i] = array[i].chars[depth];
    for (size_t i = 0; i < n; ++i) {
      assert(oracle[i] < K);
      ++bucketsize[oracle[i]];
    }
    ssize_t bucketindex[K];
    bucketindex[0] = bucketsize[0];
    size_t last_bucket_size = bucketsize[0];
    for (unsigned i = 1; i < K; ++i) {
      bucketindex[i] = bucketindex[i - 1] + bucketsize[i];
      if (bucketsize[i]) last_bucket_size = bucketsize[i];
    }
    for (size_t i = 0, j; i < n - last_bucket_size;) {
      while ((j = --bucketindex[oracle[i]]) > i) {
        std::swap(array[i], array[j]);
        std::swap(oracle[i], oracle[j]);
      }
      i += bucketsize[oracle[i]];
    }
    free(oracle);

    size_t bsum = 0;
    for (size_t i = 0; i < K; bsum += bucketsize[i++]) {
      if (bucketsize[i] <= 1) continue;
      radixsort_CI2<MaxDepth>(array + bsum, bucketsize[i], depth + 1, K);
    }
  }

  // **********************************************************************
  // *** MPI variables

  kamping::Communicator<> comm;

  static const unsigned int MSGTAG = 42;  // arbitrary number

  int samplefactor;

  pDCX() : comm(kamping::Communicator<>()) {
    samplefactor = comm.size();  // TODO
  }

  ~pDCX() {}

  template <typename Type>
  void gather_vector(
      const std::vector<Type>& v, std::vector<Type>& out,
      unsigned int removelap = 0,
      kamping::Communicator<>& comm = kamping::Communicator<>()) {
    int size = v.size() - (comm.rank() != comm.size() - 1 ? removelap : 0);
    out = comm.gatherv(kamping::send_buf(v), kamping::send_count(size));
  }

  // **********************************************************************
  // *** MPI variables

  static inline bool cmpTupleNCompare(const TupleN& t1, const TupleN& t2) {
    unsigned int v1 = t1.index % X, v2 = t2.index % X;

    const int* deprank = DCParam::cmpDepthRanks[v1][v2];

    if (debug_compare)
      std::cout << "cmp " << v1 << t1 << " ? " << v2 << t2 << " - depth "
                << deprank[0] << "\n";

    for (int d = 0; d < deprank[0]; ++d) {
      if (t1.chars[d] == t2.chars[d]) continue;
      return (t1.chars[d] < t2.chars[d]);
    }

    if (debug_compare)
      std::cout << "break tie using ranks " << deprank[1] << " - " << deprank[2]
                << " = " << t1.ranks[deprank[1]] << " - "
                << t2.ranks[deprank[2]] << "\n";

    // assert (t1.ranks[ deprank[1] ] != t2.ranks[ deprank[2] ]);

    return (t1.ranks[deprank[1]] < t2.ranks[deprank[2]]);
  }

  struct TupleNMerge {
    const std::vector<TupleN>* m_S;

    std::vector<unsigned int> m_ptr;

    TupleNMerge(const std::vector<TupleN>* S) : m_S(S), m_ptr(X, 0) {}

    inline bool done(int v) const { return (m_ptr[v] >= m_S[v].size()); }

    inline bool operator()(int v1, int v2) const {
      assert(v1 < v2);

      const int* deprank = DCParam::cmpDepthRanks[v1][v2];

      const TupleN& t1 = m_S[v1][m_ptr[v1]];
      const TupleN& t2 = m_S[v2][m_ptr[v2]];

      assert(t1.index % X == (unsigned int)v1);
      assert(t2.index % X == (unsigned int)v2);

      if (debug_compare)
        std::cout << "cmp " << v1 << "(" << t1.index << ") ? " << v2 << "("
                  << t2.index << ") - depth " << deprank[0] << "\n";

      for (int d = 0; d < deprank[0]; ++d) {
        if (t1.chars[d] == t2.chars[d]) continue;
        return (t1.chars[d] < t2.chars[d]);
      }

      if (debug_compare)
        std::cout << "break tie using ranks " << deprank[1] << " - "
                  << deprank[2] << " = " << t1.ranks[deprank[1]] << " - "
                  << t2.ranks[deprank[2]] << "\n";

      assert(t1.ranks[deprank[1]] != t2.ranks[deprank[2]]);

      return (t1.ranks[deprank[1]] < t2.ranks[deprank[2]]);
    }
  };

  // functions for rebalancing the input
  static inline uint RangeFix(uint a, uint b, uint limit) {
    if (b >= a) return 0;
    return std::min<uint>(limit, a - b);
  }

  // functions for rebalancing the input
  inline uint Extra(int i) {
    return (i != comm.size_signed() - 1) ? (X - 1) : 0;
  }

  bool dcx(std::vector<alphabet_type>& input, uint depth, uint K) {
    const unsigned int* DC = DCParam::DC;

    // **********************************************************************
    // * analyze input and rebalance input to localStride, which is a multiple
    // of p and X.

    // collect all localSizes and calc prefix sum

    unsigned int localSize = input.size();

    auto localSizes = comm.allgather(kamping::send_buf(localSize));
    localSizes.resize(comm.size() + 1);

    exclusive_prefixsum(localSizes.data(), comm.size());

    DBG_ARRAY2(debug_rebalance, "localSizes", localSizes.data(),
               comm.size() + 1);

    // calculate localStride

    const uint globalSize = localSizes[comm.size()];  // global size of input

    uint localStride = (globalSize + comm.size() - 1) /
                       comm.size();      // divide by processors rounding up
    localStride += X - localStride % X;  // round up to nearest multiple of X

    const unsigned int localOffset = comm.rank() * localStride;
    localSize =
        (comm.rank() != comm.size() - 1)
            ? localStride
            : globalSize -
                  localOffset;  // target localSize (without extra tuples)
    const unsigned int localSizeExtra =
        (comm.rank() != comm.size() - 1)
            ? localStride + (X - 1)
            : globalSize - localOffset;  // target localSize with extra tuples

    const unsigned int globalMultipleOfX =
        (globalSize + X - 1) / X;  // rounded up number of global multiples of X
    const unsigned int M =
        (localSize + X - 1) /
        X;  // number of incomplete X chars in local area size

    uint samplesize =
        (uint)sqrt(localStride * D / X / comm.size()) * samplefactor;
    if (samplesize >= D * (localStride / X))
      samplesize = D * (localStride / X) - 1;

    if (debug) {
      std::cout << "******************** DCX (process " << comm.rank()
                << ") depth " << depth << " ********************" << std::endl;

      std::cout << "Parameters:\n"
                << "  globalSize = " << globalSize << "\n"
                << "  localStride = " << localStride << "\n"
                << "  localSize = " << localSize << "\n"
                << "  localSizeExtra = " << localSizeExtra << "\n"
                << "  globalMultipleOfX = " << globalMultipleOfX << "\n"
                << "  localMultipleOfX (aka M) = " << M << "\n"
                << "  samplesize = " << samplesize << "\n"
                << "  K = " << K << "\n"
                << "  current memusage = mem " << getmemusage() << "\n";
    }

    // rebalance input

    {
      std::vector<int> sendcnt(comm.size(), 0);
      std::vector<int> sendoff(comm.size(), 0);
      std::vector<int> recvcnt(comm.size(), 0);
      std::vector<int> recvoff(comm.size(), 0);

      for (int i = 1; i < comm.size_signed(); ++i) {
        if (debug_rebalance) {
          std::cout << "range sent " << comm.rank() << " -> " << i << " is "
                    << RangeFix(i * localStride, localSizes[comm.rank()],
                                input.size())
                    << " - "
                    << RangeFix((i + 1) * localStride + Extra(i),
                                localSizes[comm.rank()], input.size())
                    << "\n";
        }

        sendoff[i] =
            RangeFix(i * localStride, localSizes[comm.rank()], input.size());
        sendcnt[i - 1] = RangeFix(i * localStride + Extra(i - 1),
                                  localSizes[comm.rank()], input.size()) -
                         sendoff[i - 1];
      }
      sendcnt[comm.size() - 1] = input.size() - sendoff[comm.size() - 1];

      DBG_ARRAY2(debug_rebalance, "sendcnt", sendcnt.data(), comm.size());
      DBG_ARRAY2(debug_rebalance, "sendoff", sendoff.data(), comm.size());

      for (int i = 1; i < comm.size_signed(); ++i) {
        if (debug_rebalance) {
          std::cout << "range recv " << i << " -> " << comm.rank() << " is "
                    << RangeFix(localSizes[i], comm.rank() * localStride,
                                localSizeExtra)
                    << "\n"
                    << RangeFix(localSizes[i + 1], comm.rank() * localStride,
                                localSizeExtra)
                    << "\n";
        }

        recvoff[i] =
            RangeFix(localSizes[i], comm.rank() * localStride, localSizeExtra);
        recvcnt[i - 1] =
            RangeFix(localSizes[i], comm.rank() * localStride, localSizeExtra) -
            recvoff[i - 1];
      }
      recvcnt[comm.size() - 1] = localSizeExtra - recvoff[comm.size() - 1];

      DBG_ARRAY2(debug_rebalance, "recvcnt", recvcnt.data(), comm.size());
      DBG_ARRAY2(debug_rebalance, "recvoff", recvoff.data(), comm.size());

      std::vector<alphabet_type> recvbuf(localSizeExtra);

      input = comm.alltoallv(
          kamping::send_buf(input), kamping::send_counts(sendcnt),
          kamping::send_displs(sendoff), kamping::recv_counts(recvcnt),
          kamping::recv_displs(recvoff));
    }

    DBG_ARRAY2(debug_input, "Input (without extra tuples)", input.data(),
               localSize);

    DBG_ARRAY2(debug_input, "Input (extra tuples)", (input.data() + localSize),
               localSizeExtra - localSize);

    // **********************************************************************
    // * calculate build DC-tuple array and sort locally

    std::vector<TupleS> R(
        D * M);  // create D * M tuples which might include up to D-1 dummies

    uint j = 0;
    for (uint i = 0; i < localSize; i += X) {
      for (uint d = 0; d < D; ++d) {
        R[j].index = localOffset + i + DC[d];

        for (uint x = i + DC[d], y = 0; y < X; ++x, ++y)
          R[j].chars[y] = (x < localSizeExtra) ? input[x] : 0;

        ++j;
      }
    }

    assert(j == D * M);

    std::cout << "done local sort sample suffixes - mem = " << getmemusage()
              << "\n";
    std::cout << "sizeof R = " << R.size() * sizeof(R[0]) << " - "
              << R.capacity() * sizeof(R[0]) << "\n";

    DBG_ARRAY(debug_sortsample, "Sample suffixes", R);

    // **********************************************************************
    // {{{ Sample sort of array R
    {
      // sort locally
      if (K < 4096)
        radixsort_CI<X>(R.data(), R.size(), 0, K);
      else
        std::sort(R.begin(), R.end());

      std::cout << "done local sort sample suffixes\n";

      DBG_ARRAY(debug_sortsample, "Locally sorted sample suffixes", R);

      // **********************************************************************
      // * select equidistance samples and redistribute sorted DC-tuples

      // select samples
      std::vector<TupleS> samplebuf(samplesize);

      double dist = (double)R.size() / samplesize;
      for (uint i = 0; i < samplesize; i++) samplebuf[i] = R[int(i * dist)];

      auto samplebufall = comm.gather(kamping::send_buf(samplebuf));
      vector_free(samplebuf);

      // root proc sorts samples as splitters

      std::vector<TupleS> splitterbuf(comm.size());

      if (comm.is_root()) {
        std::sort(samplebufall.begin(), samplebufall.end());

        DBG_ARRAY2(debug_sortsample, "Sample splitters", samplebufall.data(),
                   comm.size() * samplesize);

        for (int i = 0; i < comm.size_signed(); i++)  // pick splitters
          splitterbuf[i] = samplebufall[i * samplesize];

        DBG_ARRAY2(debug_sortsample, "Selected splitters", splitterbuf,
                   comm.size_signed());

        vector_free(samplebufall);
      }

      // distribute splitters
      comm.bcast(kamping::send_recv_buf(splitterbuf));

      // find nearest splitters in locally sorted tuple list

      std::vector<uint> splitterpos(comm.size() + 1, 0);

      splitterpos[0] = 0;
      for (int i = 1; i < comm.size_signed(); i++) {
        typename std::vector<TupleS>::const_iterator it =
            std::lower_bound(R.begin(), R.end(), splitterbuf[i]);

        splitterpos[i] = it - R.begin();
      }
      splitterpos[comm.size()] = R.size();

      DBG_ARRAY2(debug_sortsample, "Splitters positions", splitterpos.data(),
                 comm.size() + 1);

      vector_free(splitterbuf);

      // boardcast number of element in each division

      std::vector<int> sendcnt(comm.size());
      // int* recvcnt = new int[ comm.size() ];

      for (int i = 0; i < comm.size_signed(); i++) {
        sendcnt[i] = splitterpos[i + 1] - splitterpos[i];
        assert(sendcnt[i] >= 0);
      }

      vector_free(splitterpos);

      std::vector<int> recvoff(comm.size() + 1);
      std::vector<int> recvcnt(comm.size(), 0);
      R = comm.alltoallv(kamping::send_buf(R), kamping::send_counts(sendcnt),
                         kamping::recv_displs_out(recvoff),
                         kamping::recv_counts_out(recvcnt));

      recvoff[comm.size()] =
          recvcnt[comm.size() - 1] + recvoff[comm.size() - 1];

      vector_free(sendcnt);
      vector_free(recvcnt);

      merge_areas(R, recvoff.data(), comm.size());

      vector_free(recvoff);
    }
    // }}} end Sample sort of array R

    DBG_ARRAY(debug_sortsample, "Sorted sample suffixes", R);
    std::cout << "done global sort sample suffixes - mem = " << getmemusage()
              << "\n";

    std::cout << "myproc " << comm.rank() << " R.size() = " << R.size() << "\n";

    // R contains DC-sample tuples in sorted order

    // **********************************************************************
    // * Lexicographical naming

    std::vector<Pair> P(R.size());

    uint lastname, recursion;

    {
      // naming with local names

      unsigned int dupnames = 0;

      TupleS temp;  // get last tuple from previous process as basis for name
                    // comparison (cyclicly)
      comm.isend(kamping::send_buf(R.back()),
                 kamping::destination(comm.rank_shifted_cyclic(1)),
                 kamping::tag(MSGTAG));
      comm.recv(kamping::recv_buf(temp),
                kamping::source(comm.rank_shifted_cyclic(-1)),
                kamping::tag(MSGTAG));

      uint name = 0, unique = 0;
      for (uint i = 0; i < R.size(); i++) {
        if (!(R[i] == temp)) {
          name++;
          if (debug_nameing)
            std::cout << "Giving name " << name << " to " << R[i] << "\n";
          temp = R[i];
          unique = 1;
        } else {
          dupnames++;
          if (i != 0) P[i - 1].unique = 0;
          unique = 0;
        }
        P[i].name = name;
        P[i].index = R[i].index;
        P[i].unique = unique;
      }
      vector_free(
          R);  // Why?: because it is easier to recreate the tuples later on

      std::cout << "given: dupnames " << dupnames << " names given: " << name
                << " total: " << P.size() << "\n";

      DBG_ARRAY(debug_nameing, "Local Names", P);

      // renaming with global names: calculate using prefix sum

      uint namesglob = comm.scan_single(kamping::send_buf(name),
                                        kamping::op(kamping::ops::plus<>()));

      // update local names - and free up first D names for sentinel ranks
      for (uint i = 0; i < P.size(); i++) P[i].name += (namesglob - name) + D;

      DBG_ARRAY(debug_nameing, "Global Names", P);

      // determine whether recursion is necessary: last proc broadcasts highest
      // name

      if (comm.rank() == comm.size() - 1) lastname = P.back().name;

      comm.bcast(kamping::send_recv_buf(lastname),
                 kamping::root(comm.size() - 1));

      if (1 || debug_nameing)
        std::cout << "last name: " << lastname << " =? "
                  << D * globalMultipleOfX + D << "\n";

      recursion = (lastname != D * globalMultipleOfX + D);

      if (1 || debug_nameing) std::cout << "recursion: " << recursion << "\n";
    }

    std::cout << "done naming - mem = " << getmemusage() << "\n";

    if (recursion) {
      uint namesGlobalSize =
          D * globalMultipleOfX + D;  // add D dummies separating mod-X areas
      uint namesLocalStride =
          (namesGlobalSize + comm.size() - 1) / comm.size();  // rounded up
      namesLocalStride +=
          X - namesLocalStride % X;  // round up to nearest multiple of X
      uint namesGlobalMultipleOfX =
          globalMultipleOfX +
          1;  // account one extra X-tuple for D separation dummies

      std::cout << "namesGlobalSize = " << namesGlobalSize << "\n"
                << "namesLocalStride = " << namesLocalStride << "\n";

      if (comm.rank() == comm.size() - 1) {
        for (unsigned int i = 0; i < D; ++i) {
          Pair x;
          x.index = globalMultipleOfX * X + DC[i];
          x.name = D - 1 - i;
          x.unique = 1;
          P.push_back(x);
        }
      }

      if (debug_recursion) {
        std::vector<Pair> Pall;
        gather_vector(P, Pall, 0, comm);

        if (comm.is_root()) {
          std::sort(Pall.begin(), Pall.end(),
                    Pair::cmpIndexModDiv);  // sort locally

          DBG_ARRAY(debug_recursion, "Pall", Pall);
        }
      }

      if (namesGlobalSize > 2 * X * comm.size()) {
        if (debug_recursion)
          std::cout << "---------------------   RECURSION pDCX "
                       "----------------  - mem = "
                    << getmemusage() << std::endl;

        // **********************************************************************
        // {{{ Sample sort of array P by (i mod X, i div X)

        std::sort(P.begin(), P.end(), Pair::cmpIndexModDiv);  // sort locally

        DBG_ARRAY(debug_recursion, "Names locally sorted by cmpModDiv", P);

        std::vector<uint> splitterpos(comm.size() + 1, 0);
        std::vector<int> sendcnt(comm.size(), 0);

        // use equidistance splitters from 0..namesGlobalSize (because indexes
        // are known in advance)
        splitterpos[0] = 0;
        Pair ptemp;
        ptemp.name = 0;
        for (int i = 1; i < comm.size_signed(); i++) {
          ptemp.index = i * namesLocalStride;

          unsigned int x = ptemp.index;

          unsigned int divM = ptemp.index / namesGlobalMultipleOfX;
          ptemp.index =
              DC[divM] + X * (ptemp.index - divM * namesGlobalMultipleOfX);

          if (debug_recursion)
            std::cout << "splitter: " << ptemp.index << " = " << x << " - "
                      << divM << "\n";

          typename std::vector<Pair>::const_iterator it =
              std::lower_bound(P.begin(), P.end(), ptemp, Pair::cmpIndexModDiv);
          splitterpos[i] = it - P.begin();
        }
        splitterpos[comm.size()] = P.size();

        DBG_ARRAY2(debug_recursion, "Splitters positions", splitterpos,
                   comm.size() + 1);

        for (int i = 0; i < comm.size_signed(); i++) {
          sendcnt[i] = splitterpos[i + 1] - splitterpos[i];
          assert(sendcnt[i] >= 0);
        }

        std::vector<int> recvcnt(comm.size());
        std::vector<int> recvoff(comm.size() + 1);
        std::vector<Pair> recvBufPair =
            comm.alltoallv(kamping::send_buf(P), kamping::send_counts(sendcnt),
                           kamping::recv_displs_out(recvoff),
                           kamping::recv_counts_out(recvcnt));
        recvoff[comm.size()] =
            recvoff[comm.size() - 1] + recvcnt[comm.size() - 1];

        vector_free(P);

        // final X-1 tuples should be ignored due to recvoff areas
        merge_areas(recvBufPair, recvoff.data(), comm.size(),
                    Pair::cmpIndexModDiv);

        // TODO: merge and reduce at once

        uint uniqueseq = 0;

        std::vector<uint> namearray(recvBufPair.size());
        for (unsigned int i = 0; i < recvBufPair.size(); ++i) {
          if (i != 0) {
            if (recvBufPair[i - 1].unique && recvBufPair[i].unique) uniqueseq++;
          }

          namearray[i] = recvBufPair[i].name;
        }

        DBG_ARRAY(debug_recursion, "Pairs P (globally sorted by indexModDiv)",
                  recvBufPair);

        std::cout << "uniques in sequence: " << uniqueseq << " - "
                  << recvBufPair.size() / 2 << "\n";

        // }}} end Sample sort of array P

        if (uniqueseq > recvBufPair.size() / 2 && 0) {
          // **********************************************************************
          // * recurse on compressed sequence of duplicates and uniques

          // reuse name array's second half for indexes
          uint* indexarray = namearray.data() + recvBufPair.size() / 2;

          uint j = 0;
          for (unsigned int i = 0; i < recvBufPair.size(); ++i) {
            if (i != 0) {
              if (recvBufPair[i - 1].unique && recvBufPair[i].unique) continue;
            }

            namearray[j] = recvBufPair[i].name;

            unsigned int divM = i / namesGlobalMultipleOfX;
            uint index = DC[divM] + X * (i - divM * namesGlobalMultipleOfX);

            indexarray[j] = index;
            std::cout << "dup/firstunique name: " << namearray[j] << " - index "
                      << indexarray[j] << "\n";
            ++j;
          }

          uint oldNamesGlobalSize = namesGlobalSize;
          namesGlobalSize = j;
          namesLocalStride =
              (namesGlobalSize + comm.size() - 1) / comm.size();  // rounded up
          namesLocalStride +=
              X - namesLocalStride % X;  // round up to nearest multiple of X

          assert(j < recvBufPair.size() / 2);

          vector_free(recvBufPair);

          pDCX<DCParam, uint> rdcx;
          // rdcx.dcx( namearray.data(), namesGlobalSize - (X-1),
          // namesLocalStride, depth+1, oldNamesGlobalSize+1 );

          std::cout << "SAlocal: " << rdcx.localSA.size() << " - indexes " << j
                    << "\n";
          std::cout << "SAlocal: " << rdcx.localSA.size() << " - indexes " << j
                    << "\n";
          std::cout << "SAlocal: " << rdcx.localSA.size() << " - indexes " << j
                    << "\n";
          std::cout << "SAlocal: " << rdcx.localSA.size() << " - indexes " << j
                    << "\n";

          assert(0);
        } else {
          // recurse on full sequence of names

          vector_free(recvBufPair);

          DBG_ARRAY(debug_recursion, "namearray", namearray);

          assert(namearray.size() == namesLocalStride ||
                 comm.rank() == comm.size() - 1);

          pDCX<DCParam, uint> rdcx;
          rdcx.dcx(namearray, depth + 1, lastname + 1);

          if (debug_rec_selfcheck) {
            if (debug)
              std::cout << "---------------------   RECURSION local checkSA "
                           "---------------- "
                        << localSize << std::endl;

            std::vector<uint> nameAll;
            std::vector<uint> SAall;

            gather_vector(namearray, nameAll, X - 1, comm);
            gather_vector(rdcx.localSA, SAall, 0, comm);

            DBG_ARRAY(debug_recursion, "nameAll", nameAll);
            DBG_ARRAY(debug_recursion, "SAall", SAall);

            if (comm.is_root()) {
              assert(sachecker::sa_checker(nameAll, SAall));
            }

            comm.barrier();
          }

          vector_free(namearray);

          DBG_ARRAY(debug_recursion, "Recursive localSA", rdcx.localSA);

          uint SAsize = rdcx.localSA.size();
          std::vector<uint> allSAsize(comm.size() + 1, 0);

          comm.allgather(kamping::send_buf(SAsize),
                         kamping::recv_buf(allSAsize));

          exclusive_prefixsum(allSAsize.data(), comm.size());

          DBG_ARRAY2(debug_recursion, "allSAsize", allSAsize.data(),
                     comm.size() + 1);

          // generate array of pairs (index,rank) from localSA

          P.resize(rdcx.localSA.size());

          for (unsigned int i = 0; i < rdcx.localSA.size(); ++i) {
            // generate index in ModDiv sorted input sequence

            uint saidx = rdcx.localSA[i];

            unsigned int divM = saidx / namesGlobalMultipleOfX;

            uint index = DC[divM] + X * (saidx - divM * namesGlobalMultipleOfX);

            P[i].index = index;
            P[i].name = allSAsize[comm.rank()] + i + 1;
          }
        }
      } else  // use sequential suffix sorter
      {
        if (debug)
          std::cout << "---------------------   RECURSION local sais "
                       "---------------- "
                    << localSize << std::endl;

        std::vector<Pair> Pall;
        gather_vector(P, Pall, 0, comm);

        if (comm.is_root()) {
          assert(Pall.size() == (int)namesGlobalSize);

          DBG_ARRAY(debug_recursion, "Global Names sorted index", Pall);

          std::sort(Pall.begin(), Pall.end(),
                    Pair::cmpIndexModDiv);  // sort locally

          DBG_ARRAY(debug_recursion, "Global Names sorted cmpModDiv", Pall);

          uint* namearray = new uint[Pall.size()];
          for (unsigned int i = 0; i < Pall.size(); ++i)
            namearray[i] = Pall[i].name;

          DBG_ARRAY2(debug_recursion, "Recursion input", namearray,
                     Pall.size());

          int* rSA = new int[Pall.size()];

          yuta_sais_lite::saisxx<uint*, int*, int>(namearray, rSA, Pall.size(),
                                                   lastname + 1);

          delete[] namearray;

          DBG_ARRAY2(debug_recursion, "Recursive SA", rSA, Pall.size());

          // generate rank array - same as updating pair array with correct
          // names
          for (uint i = D; i < Pall.size(); ++i) {
            Pall[rSA[i]].name = i + D;
          }

          DBG_ARRAY(debug_recursion, "Fixed Global Names sorted cmpModDiv",
                    Pall);

          std::swap(P, Pall);
        } else {
          vector_free(P);
        }

      }  // end use sequential suffix sorter

      if (debug_recursion)
        std::cout << "---------------------   END  RECURSION  ---------------  "
                     "- mem = "
                  << getmemusage() << std::endl;
    } else {
      if (debug_recursion)
        std::cout << "---------------------   keine  Recursion---------------- "
                     "- mem = "
                  << getmemusage() << std::endl;
    }

    // in any outcome: here P contains pairs of index and unique rank.

    {
      // **********************************************************************
      // *** sample sort pairs P by index

      std::sort(P.begin(), P.end());

      DBG_ARRAY(debug_recursion, "Pairs P (sorted by index)", P);

      std::vector<uint> splitterpos(comm.size() + 1, 0);
      std::vector<int> sendcnt(comm.size(), 0);

      // use equidistance splitters from 0..globalSize (because names are
      // unique)
      splitterpos[0] = 0;
      Pair ptemp;
      ptemp.name = 0;
      for (int i = 1; i < comm.size_signed(); i++) {
        ptemp.index = i * localStride;

        typename std::vector<Pair>::const_iterator it =
            std::lower_bound(P.begin(), P.end(), ptemp);
        splitterpos[i] = it - P.begin();
      }
      splitterpos[comm.size()] = P.size();

      for (int i = 0; i < comm.size_signed(); i++) {
        sendcnt[i] = splitterpos[i + 1] - splitterpos[i];
        assert(sendcnt[i] >= 0);
      }

      std::vector<int> recvcnt(comm.size(), 0);
      std::vector<int> recvoff(comm.size() + 1, 0);
      std::vector<Pair> recvBufPair = comm.alltoallv(
          kamping::send_buf(P), kamping::send_counts(sendcnt),
          kamping::recv_counts_out(recvcnt), kamping::recv_displs_out(recvoff));
      recvoff[comm.size()] =
          recvoff[comm.size() - 1] + recvcnt[comm.size() - 1];
      unsigned int recvBufPairSize = recvoff[comm.size()];
      recvBufPair.resize(recvoff[comm.size()] + D);

      vector_free(P);

      merge_areas(recvBufPair, recvoff.data(), comm.size());

      DBG_ARRAY2(debug_recursion, "Pairs P (globally sorted by index)",
                 recvBufPair.data(), recvBufPairSize);

      // **********************************************************************
      // *** every PE needs D additional sample suffix ranks to complete the
      // final tuple

      std::vector<Pair> temp(D);

      comm.isend(kamping::send_buf(recvBufPair), kamping::send_count(D),
                 kamping::destination(comm.rank_shifted_cyclic(-1)),
                 kamping::tag(MSGTAG));
      comm.recv(kamping::recv_buf(temp), kamping::recv_count(D),
                kamping::source(comm.rank_shifted_cyclic(1)),
                kamping::tag(MSGTAG));

      DBG_ARRAY2(debug_recursion, "Pairs P (extra tuples)", temp, D);

      if (comm.rank() ==
          comm.size() -
              1)  // last processor gets sentinel tuples with lowest ranks
      {
        for (unsigned int i = 0; i < D; ++i) {
          // the first D ranks are freed up (above) for the following sentinel
          // ranks 0..D-1:
          recvBufPair[recvBufPairSize + i].name = D - i - 1;
          recvBufPair[recvBufPairSize + i].index =
              recvBufPair[recvBufPairSize - D].index - DC[0] + X + DC[i];
        }
      } else  // other processors get D following tuples with indexes from the
              // DC
      {
        for (unsigned int i = 0; i < D; ++i) {
          recvBufPair[recvBufPairSize + i].name = temp[i].name;
          recvBufPair[recvBufPairSize + i].index =
              recvBufPair[recvBufPairSize - D].index - DC[0] + X + DC[i];
          assert(recvBufPair[recvBufPairSize + i].index == temp[i].index);
        }
      }

      std::swap(recvBufPair, P);

      DBG_ARRAY(debug_recursion,
                "Pairs P (globally sorted by index + extra tuples)", P);
    }

    // P contains pairs of index and global rank: sorted by index and
    // partitioned by localSize

    // **********************************************************************
    // *** Generate tuple arrays of samples and non-samples

    std::vector<TupleN> S[X];

    for (unsigned int k = 0; k < X; ++k) S[k].resize(M);

    unsigned int dp =
        0;  // running index into P incremented when crossing DC-indexes

    for (unsigned int i = 0; i < M; ++i) {
      for (unsigned int k = 0; k < X; ++k) {
        S[k][i].index = comm.rank() * localStride + i * X + k;

        for (unsigned int c = 0; c < X - 1; ++c)
          S[k][i].chars[c] =
              (i * X + k + c < localSizeExtra) ? input[i * X + k + c] : 0;

        for (unsigned int d = 0; d < D; ++d) S[k][i].ranks[d] = P[dp + d].name;

        if (DC[dp % D] == k) ++dp;
      }
    }

    std::cout << "done creating S_i's - mem = " << getmemusage() << "\n";

    // **********************************************************************
    // *** Sample sort tuple arrays
    {
      for (unsigned int k = 0; k < X; ++k) {
        // TODO: sort less - not all S[k] must be sorted to depth X-1 (needs
        // additional lookup table)
        if (K < 4096)
          radixsort_CI2<X - 1>(S[k].data(), S[k].size(), 0, K);
        else
          std::sort(S[k].begin(), S[k].end(), cmpTupleNdepth<X - 1>);
      }

      // select equidistant samples

      std::vector<TupleN> samplebuf(X * samplesize);

      double dist = (double)M / samplesize;
      for (uint i = 0, j = 0; i < samplesize; i++) {
        for (unsigned int k = 0; k < X; ++k) {
          samplebuf[j++] = S[k][int(i * dist)];
        }
      }

      std::vector<TupleN> samplebufall =
          comm.gather(kamping::send_buf(samplebuf));

      vector_free(samplebuf);

      // root proc sorts samples as splitters

      std::vector<TupleN> splitterbuf(comm.size());

      if (comm.is_root()) {
        std::sort(samplebufall.begin(), samplebufall.end(), cmpTupleNCompare);

        DBG_ARRAY2(debug_finalsort, "Sample splitters", samplebufall.data(),
                   comm.size() * X * samplesize);

        for (int i = 0; i < comm.size_signed(); i++)  // pick splitters
          splitterbuf[i] = samplebufall[i * X * samplesize];

        DBG_ARRAY2(debug_finalsort, "Selected splitters", splitterbuf,
                   comm.size());

        vector_free(samplebufall);
      }

      // distribute splitters
      comm.bcast(kamping::send_recv_buf(splitterbuf));

      // find nearest splitters in each of the locally sorted tuple list

      uint** splitterpos = new uint*[X];

      for (unsigned int k = 0; k < X; ++k) {
        splitterpos[k] = new uint[comm.size() + 1];

        splitterpos[k][0] = 0;
        for (int i = 1; i < comm.size_signed(); i++) {
          typename std::vector<TupleN>::const_iterator it = std::lower_bound(
              S[k].begin(), S[k].end(), splitterbuf[i], cmpTupleNCompare);

          splitterpos[k][i] = it - S[k].begin();
        }
        splitterpos[k][comm.size()] = M;
      }

      for (unsigned int k = 0; k < X; ++k) {
        DBG_ARRAY2(debug_finalsort, "Splitters S." << k, splitterpos[k],
                   comm.size() + 1);
      }

      vector_free(splitterbuf);

      // boardcast number of element in each division

      // int** sendcnt = new int*[X];
      std::vector<std::vector<int>> sendcnt(X,
                                            std::vector<int>(comm.size(), 0));
      std::vector<std::vector<int>> recvcnt(X,
                                            std::vector<int>(comm.size(), 0));
      std::vector<std::vector<int>> recvoff(
          X, std::vector<int>(comm.size() + 1, 0));

      for (unsigned int k = 0; k < X; ++k) {
        for (int i = 0; i < comm.size_signed(); i++) {
          sendcnt[k][i] = splitterpos[k][i + 1] - splitterpos[k][i];

          assert(sendcnt[k][i] >= 0);
        }

        delete[] splitterpos[k];
      }

      delete[] splitterpos;

      // calculate number of received tuples
      unsigned int totalsize = 0;

      for (unsigned int k = 0; k < X; ++k) {
        S[k] = comm.alltoallv(kamping::send_buf(S[k]),
                              kamping::send_counts(sendcnt[k]),
                              kamping::recv_counts_out(recvcnt[k]),
                              kamping::recv_displs_out(recvoff[k]));

        recvoff[k][comm.size()] =
            recvoff[k][comm.size() - 1] + recvcnt[k][comm.size() - 1];

        totalsize += S[k].size();
      }

      vector_free(sendcnt);
      vector_free(recvcnt);

      // merge received array parts

      for (unsigned int k = 0; k < X; ++k) {
        if (S[k].size()) {
          merge_areas(S[k], recvoff[k].data(), comm.size(),
                      cmpTupleNdepth<X - 1>);
        }
      }

      vector_free(recvoff);

      for (unsigned int k = 0; k < X; ++k) {
        DBG_ARRAY(debug_finalsort, "After samplesort S" << k, S[k]);
      }

      std::vector<TupleN> suffixarray(totalsize);
      localSA.resize(totalsize);
      int j = 0;

      TupleNMerge tuplecmp(S);
      LoserTree<TupleNMerge> LT(X, tuplecmp);

      int top;

      while ((top = LT.top()) >= 0) {
        suffixarray[j] = S[top][tuplecmp.m_ptr[top]];
        localSA[j] = suffixarray[j].index;

        if (debug_finalsort)
          std::cout << "Winning tuple: " << suffixarray[j] << "\n";

        if (suffixarray[j].index < globalSize) ++j;

        tuplecmp.m_ptr[top]++;

        LT.replay();
      }

      DBG_ARRAY2(debug_finalsort, "Suffixarray merged", suffixarray, j);

      std::cout << "done merging suffixarray - mem = " << getmemusage() << "\n";

      localSA.resize(j);
    }

    if (debug) {
      std::cout << "******************** finished DCX (process " << comm.rank()
                << ") depth " << depth << " ********************" << std::endl;
    }

    return true;
  }

  uint globalSize;

  uint localStride;

  std::vector<uint> localSA;

  std::vector<uint8_t> localInput;

  bool run(const char* filename, kamping::Communicator<>& comm) {
    // **********************************************************************
    // * Read input file size

    if (comm.is_root()) {
      std::ifstream infile(filename);

      if (!infile.good()) {
        perror("Cannot read input file");
        return false;
      }

      // determine file size
      infile.seekg(0, std::ios::end);
      globalSize = infile.tellg();

      char* endptr = NULL;
      unsigned int reducesize =
          getenv("SIZE") ? strtoul(getenv("SIZE"), &endptr, 10) : 0;
      if (!endptr || *endptr != '\0') reducesize = 0;

      if (reducesize && globalSize > reducesize) globalSize = reducesize;
    }

    comm.bcast_single(kamping::send_recv_buf(globalSize));

    // **********************************************************************
    // * Calculate local input size (= general stride)

    localStride = (globalSize + comm.size() - 1) /
                  comm.size();           // divide by processors rounding up
    localStride += X - localStride % X;  // round up to nearest multiple of X

    assert(localStride * comm.size() >= globalSize);

    if (comm.is_root()) {
      std::cout << "Total input size = " << globalSize
                << " bytes. localStride = " << localStride << std::endl;
    }

    // **********************************************************************
    // * Read input file and send to other processors

    localInput.resize(localStride);

    assert(sizeof(alphabet_type) == 1);

    if (comm.is_root()) {
      std::ifstream infile(filename, std::ios::binary);

      // read input for processors 1 to n-1
      for (int p = 1; p < comm.size_signed(); ++p) {
        infile.seekg(p * localStride, std::ios::beg);

        uint readsize = (p != comm.size_signed() - 1)
                            ? localStride
                            : globalSize - (p * localStride);

        std::cout << "Read for process " << p << " from pos " << p * localStride
                  << " of length " << readsize << std::endl;

        infile.read((char*)localInput.data(), readsize);
        comm.send(kamping::send_buf(localInput), kamping::destination(p),
                  kamping::send_count(readsize), kamping::tag(MSGTAG));
      }

      if (!infile.good()) {
        perror("Error reading file.");
        return false;
      }

      // read input for processor 0 (ROOT)

      std::cout << "Read for process 0 from pos 0 of length " << localStride
                << std::endl;

      infile.seekg(0, std::ios::beg);
      infile.read((char*)localInput.data(), localStride);

      if (!infile.good()) {
        perror("Error reading file.");
        return false;
      }
    } else  // not ROOT: receive data
    {
      comm.recv(kamping::recv_buf<kamping::resize_to_fit>(localInput),
                kamping::source(comm.root()), kamping::tag(MSGTAG));
    }

    comm.barrier();

    // **********************************************************************
    // * Construct suffix array recursively

    dcx(localInput, 0, 256);

    return true;
  }

  bool writeSA(const char* filename) {
    std::vector<uint> allSAsize =
        comm.gather(kamping::send_buf(static_cast<uint>(localSA.size())));

    if (comm.is_root()) {
      int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);

      if (fd < 0) {
        std::cout << "Error opening file: " << strerror(errno) << std::endl;
        return false;
      }

      // write data portion from the ROOT process
      write(fd, localSA.data(), allSAsize[0] * sizeof(uint));

      std::cout << "Wrote data from process 0." << std::endl;

      // receive data from other processes

      uint maxsize = *std::max_element(allSAsize.begin(), allSAsize.end());
      std::vector<uint> buffer(maxsize);
      for (int p = 1; p < comm.size_signed(); p++) {
        comm.recv(kamping::recv_buf(buffer), kamping::source(p),
                  kamping::tag(MSGTAG));

        ssize_t wb = write(fd, buffer.data(), allSAsize[p] * sizeof(uint));

        if ((uint)wb != allSAsize[p] * sizeof(uint)) {
          std::cout << "Error writing to file: " << strerror(errno)
                    << std::endl;
          return false;
        }

        std::cout << "Wrote data from process " << p << "." << std::endl;
      }

      if (close(fd) != 0) {
        std::cout << "Error writing to file: " << strerror(errno) << std::endl;
        return false;
      }
    } else {
      comm.send(kamping::send_buf(localSA), kamping::destination(comm.root()),
                kamping::tag(MSGTAG));
    }

    return true;
  }

  bool checkSA(kamping::Communicator<>& comm) {
    if (debug) {
      std::cout << "******************** SAChecker (process " << comm.rank()
                << ") ********************" << std::endl;
      std::cout << "localStride = " << localStride << "\n";
      std::cout << "localSA.size() = " << localSA.size() << "\n";
      std::cout << "localInput.size() = " << localInput.size() << "\n";
    }

    assert(localStride + (X - 1) == localInput.size() ||
           comm.rank() == comm.size() - 1);

    // **********************************************************************
    // * Generate pairs (SA[i],i)

    uint localSAsize = localSA.size();

    localSAsize = comm.scan_single(kamping::send_buf(localSAsize),
                                   kamping::op(kamping::ops::plus<>()));

    uint indexStart = localSAsize - localSA.size();

    std::vector<Pair> P(localSA.size());

    for (uint i = 0; i < localSA.size(); ++i) {
      P[i].index = localSA[i];
      P[i].name = indexStart + i;
    }

    DBG_ARRAY(debug_checker1, "(SA[i],i)", P);

    // **********************************************************************
    // * Sample sort of array P by (SA[i])
    {
      std::sort(P.begin(), P.end());

      uint* splitterpos = new uint[comm.size() + 1];
      std::vector<int> sendcnt(comm.size());

      // use equidistance splitters from 0..globalSize (because indexes are
      // known in advance)
      splitterpos[0] = 0;
      Pair ptemp;
      ptemp.name = 0;
      for (int i = 1; i < comm.size_signed(); i++) {
        ptemp.index = i * localStride;

        typename std::vector<Pair>::const_iterator it =
            std::lower_bound(P.begin(), P.end(), ptemp);
        splitterpos[i] = it - P.begin();
      }
      splitterpos[comm.size()] = P.size();

      DBG_ARRAY2(debug_checker1, "Splitters positions", splitterpos,
                 comm.size() + 1);

      for (int i = 0; i < comm.size_signed(); i++) {
        sendcnt[i] = splitterpos[i + 1] - splitterpos[i];
        assert(sendcnt[i] >= 0);
      }

      std::vector<int> recvcnt(comm.size());
      std::vector<int> recvoff(comm.size() + 1);
      std::vector<Pair> recvBufPair = comm.alltoallv(
          kamping::send_buf(P), kamping::send_counts(sendcnt),
          kamping::recv_counts_out(recvcnt), kamping::recv_displs_out(recvoff));
      recvoff[comm.size()] =
          recvoff[comm.size() - 1] + recvcnt[comm.size() - 1];
      recvBufPair.resize(recvBufPair.size() + 1);
      unsigned int recvBufPairSize = recvBufPair.size() - 1;

      vector_free(P);

      merge_areas(recvBufPair, recvoff.data(), comm.size());

      // **********************************************************************
      // *** every P needs 1 additional pair

      Pair temp;

      comm.isend(kamping::send_buf(recvBufPair.front()),
                 kamping::destination(comm.rank_shifted_cyclic(-1)),
                 kamping::tag(MSGTAG));
      comm.recv(kamping::recv_buf(temp),
                kamping::source(comm.rank_shifted_cyclic(1)),
                kamping::tag(MSGTAG));

      if (comm.rank() ==
          comm.size() - 1)  // last processor gets sentinel pair: virtual pair
                            // of '$' position after string
      {
        recvBufPair[recvBufPairSize].name = globalSize;
        recvBufPair[recvBufPairSize].index = INT_MAX;
      } else  // other processors get 1 following pair with indexes from the DC
      {
        recvBufPair[recvBufPairSize] = temp;
      }

      std::swap(recvBufPair, P);
    }

    // now consider P as [ (i,ISA[i]) ]_{i=0..n-1} (by substituting i -> ISA[i])

    DBG_ARRAY(
        debug_checker1,
        "(SA[i],i) sorted by SA[i] equiv: (i,ISA[i]) including 1 extra pair",
        P);

    // **********************************************************************
    // * First check: is [P.name] the sequence [0..n)

    int error = false;

    for (uint i = 0; i < P.size() - 1; ++i)  // -1 due to extra pair at end
    {
      if (P[i].index != comm.rank() * localStride + i) {
        std::cout << "SA is not a permutation of [0,n) at position "
                  << P[i].name << "\n";
        error = true;
        break;
      }
    }

    error = comm.allreduce_single(kamping::send_buf(error),
                                  kamping::op(kamping::ops::logical_or<>()));

    if (error) return false;

    // **********************************************************************
    // * Generate triples (ISA[i], ISA[i+1], S[i])

    std::vector<Triple> S(P.size() - 1);  // -1 due to extra pair at end

    for (uint i = 0; i < P.size() - 1; ++i) {
      S[i].rank1 = P[i].name;
      S[i].rank2 = P[i + 1].name;

      S[i].char1 = localInput[i];
    }

    DBG_ARRAY(debug_checker2, "(ISA[i], ISA[i+1], S[i])", S);

    // **********************************************************************
    // * Sample sort of array S by (S[].rank1)
    {
      std::sort(S.begin(), S.end());

      uint* splitterpos = new uint[comm.size() + 1];
      std::vector<int> sendcnt(comm.size());

      // use equidistance splitters from 0..globalSize (because indexes are
      // known in advance)
      splitterpos[0] = 0;
      Triple ptemp;
      for (int i = 1; i < comm.size_signed(); i++) {
        ptemp.rank1 = i * localStride;

        typename std::vector<Triple>::const_iterator it =
            std::lower_bound(S.begin(), S.end(), ptemp);
        splitterpos[i] = it - S.begin();
      }
      splitterpos[comm.size()] = S.size();

      DBG_ARRAY2(debug_checker2, "Splitters positions", splitterpos,
                 comm.size() + 1);

      for (int i = 0; i < comm.size_signed(); i++) {
        sendcnt[i] = splitterpos[i + 1] - splitterpos[i];
        assert(sendcnt[i] >= 0);
      }

      std::vector<int> recvcnt(comm.size());
      std::vector<int> recvoff(comm.size() + 1);
      std::vector<Triple> recvBuf = comm.alltoallv(
          kamping::send_buf(S), kamping::send_counts(sendcnt),
          kamping::recv_counts_out(recvcnt), kamping::recv_displs_out(recvoff));
      recvBuf.resize(recvBuf.size() + 1);
      unsigned int recvBufSize = recvBuf.size() - 1;
      recvoff[comm.size()] =
          recvoff[comm.size() - 1] + recvcnt[comm.size() - 1];

      vector_free(S);

      merge_areas(recvBuf, recvoff.data(), comm.size());

      // **********************************************************************
      // *** every P needs 1 additional triple

      Triple temp;

      comm.isend(kamping::send_buf(recvBuf.front()),
                 kamping::destination(comm.rank_shifted_cyclic(-1)),
                 kamping::tag(MSGTAG));
      comm.recv(kamping::recv_buf(temp),
                kamping::source(comm.rank_shifted_cyclic(1)),
                kamping::tag(MSGTAG));

      if (comm.rank() ==
          comm.size() - 1)  // last processor gets sentinel triple - which
                            // shouldnt be compared later on.
      {
        recvBuf[recvBufSize].rank1 = INT_MAX;
        recvBuf[recvBufSize].rank2 = INT_MAX;
        recvBuf[recvBufSize].char1 = 0;
      } else  // other processors get 1 following pair with indexes from the DC
      {
        recvBuf[recvBufSize] = temp;
      }

      std::swap(recvBuf, S);
    }

    DBG_ARRAY(debug_checker2,
              "(ISA[i], ISA[i+1], S[i]) sorted by ISA[i]\nequiv: (ISA[SA[i]], "
              "ISA[SA[i]+1], S[SA[i]]) sorted by i",
              S);

    // now consider S as [ (i, ISA[SA[i]+1], S[SA[i]]) ]_{i=0..n-1} (by
    // substituting i -> SA[i])

    // **********************************************************************
    // * Second check: use ISA to check suffix of suffixes for correct order

    unsigned int iend = S.size() - 1 - (comm.rank() == comm.size() - 1 ? 1 : 0);

    for (uint i = 0; !error && i < iend; ++i)  // -1 due to extra pair at end
    {
      if (S[i].char1 > S[i + 1].char1) {
        // simple check of first character of suffix
        std::cout << "Error: suffix array position "
                  << i + comm.rank() * localStride << " ordered incorrectly.\n";
        error = true;
      } else if (S[i].char1 == S[i + 1].char1) {
        if (S[i + 1].rank2 == globalSize) {
          // last suffix of string must be first among those
          // with same first character
          std::cout << "Error: suffix array position "
                    << i + comm.rank() * localStride
                    << " ordered incorrectly.\n";
          error = true;
        }
        if (S[i].rank2 != globalSize && S[i].rank2 > S[i + 1].rank2) {
          // positions SA[i] and SA[i-1] has same first
          // character but their suffixes are ordered
          // incorrectly: the suffix position of SA[i] is given
          // by ISA[SA[i]]
          std::cout << "Error: suffix array position "
                    << i + comm.rank() * localStride
                    << " ordered incorrectly.\n";
          error = true;
        }
      }
    }
    error = comm.allreduce_single(kamping::send_buf(error),
                                  kamping::op(kamping::ops::logical_or<>()));

    return (error == false);
  }

  bool checkSAlocal(const char* filename) {
    // **********************************************************************
    // * Read input file size

    uint globalSize;

    if (comm.is_root()) {
      std::ifstream infile(filename);

      if (!infile.good()) {
        perror("Cannot read input file");
        return false;
      }

      // determine file size
      infile.seekg(0, std::ios::end);
      globalSize = infile.tellg();

      char* endptr = NULL;
      unsigned int reducesize =
          getenv("SIZE") ? strtoul(getenv("SIZE"), &endptr, 10) : 0;
      if (!endptr || *endptr != '\0') reducesize = 0;

      if (reducesize && globalSize > reducesize) globalSize = reducesize;
    }

    comm.bcast_single(kamping::send_recv_buf(globalSize));

    // **********************************************************************
    // * Collect and check suffix array

    std::vector<uint> gSA = comm.gatherv(kamping::send_buf(localSA));

    if (comm.is_root()) {
      // DBG_ARRAY(debug_output, "Suffixarray collected", gSA);

      std::ifstream infile(filename);
      if (!infile.good()) {
        perror("Cannot read input file");
        return -1;
      }

      std::vector<uint8_t> string(globalSize);

      infile.read((char*)string.data(), globalSize);
      if (!infile.good()) {
        perror("Cannot read input file");
        return -1;
      }

      if (debug_output) {
        std::cout << "result suffix array: \n";

        for (unsigned int i = 0; i < gSA.size(); ++i) {
          std::cout << i << " : " << gSA[i] << " : ";

          for (unsigned int j = 0; gSA[i] + j < globalSize && j < 32; ++j) {
            std::cout << strC(string[gSA[i] + j]) << " ";
          }

          std::cout << "\n";
        }
      }

      assert(sachecker::sa_checker(string, gSA));
    }
    return true;
  }
};

template <typename DCParam>
void DCXRun(const char* input, const char* output) {
  kamping::Communicator<> comm(MPI_COMM_WORLD);

  int myproc = comm.rank();

  double tstart = MPI_Wtime();

  pDCX<DCParam, uint8_t> dcx;

  dcx.run(input, comm);

  if (output) {
    dcx.writeSA(output);
  }

  comm.barrier();

  double tend = MPI_Wtime();

  if (myproc == 0) {
    std::cerr << "RESULT"
              << " algo=pDC" << DCParam::X << " time=" << (tend - tstart)
              << (getenv("RESULT") ? getenv("RESULT") : "") << std::endl;
  }

  std::cout << "Suffix array checker: " << dcx.checkSA(comm) << "\n";

  dcx.checkSAlocal(input);
}

int main(int argc, char** argv) {
  kamping::Environment env;
  if (kamping::world_size() <= 1) {
    std::cerr << "Error: requires more than one MPI processor (use -np 2)."
              << std::endl;
    return -1;
  }

  if (argc < 3) {
    std::cout << "No input file! "
              << "Call using mpirun -np 4 ./pDCX <3/7/13> <input> [output]"
              << std::endl;
    return 0;
  }

  if (strcmp(argv[1], "3") == 0) {
    DCXRun<DC3Param>(argv[2], argv[3]);
  } else if (strcmp(argv[1], "7") == 0) {
    DCXRun<DC7Param>(argv[2], argv[3]);
  } else if (strcmp(argv[1], "13") == 0) {
    DCXRun<DC13Param>(argv[2], argv[3]);
  } else {
    std::cout << "Usage: pDCX <3/7/13> <input> [output]" << std::endl;
  }

  return 0;
}
