#define DBG_LIMIT 100

#define DBG_ARRAY(dbg, text, X)                                          \
  do {                                                                   \
    if (dbg) {                                                           \
      std::cout << text << " line " << __LINE__                          \
                << " : i - " #X "[i] - total size " << X.size() << "\n"; \
      for (unsigned int i = 0; i < X.size() && i < DBG_LIMIT; ++i)       \
        std::cout << i << " : " << X[i] << "\n";                         \
    }                                                                    \
  } while (0)

#define DBG_ARRAY2(dbg, text, X, Xsize)                                    \
  do {                                                                     \
    if (dbg) {                                                             \
      std::cout << text << " line " << __LINE__                            \
                << " : i - " #X "[i] - total size " << Xsize << "\n";      \
      for (unsigned int i = 0; i < (unsigned int)(Xsize) && i < DBG_LIMIT; \
           ++i)                                                            \
        std::cout << i << " : " << X[i] << "\n";                           \
    }                                                                      \
  } while (0)

unsigned int getmemusage() {
  //    struct proc_t usage;
  //    look_up_our_self(&usage);
  //    return usage.vsize;
  return 0;
}

// **********************************************************************
// * Loser tree implementation

template <typename Comparator>
class LoserTree {
 private:
  /// the tree of size n-1
  std::vector<int> m_tree;

  /// the comparator object of this tree
  const Comparator& m_less;

 public:
  LoserTree(unsigned int size, const Comparator& less) : m_less(less) {
    // initialize loser tree by performing comparisons

    unsigned int treesize = (1 << (int)(log2(size - 1) + 2)) - 1;
    m_tree.resize(treesize, -1);

    // fill in lowest level: ascending numbers until each sequence
    // finishes.
    int levelput = m_tree.size() / 2;
    for (unsigned int i = 0; i < size; ++i)
      m_tree[levelput + i] = m_less.done(i) ? -1 : i;

    int levelsize = levelput + 1;

    // construct higher levels iteratively from bottom up
    while (levelsize > 1) {
      levelsize = levelsize / 2;
      int levelget = levelput;
      levelput /= 2;

      for (int i = 0; i < levelsize; ++i) {
        if (m_tree[levelget + 2 * i + 1] < 0)
          m_tree[levelput + i] = m_tree[levelget + 2 * i];
        else if (m_tree[levelget + 2 * i] < 0)
          m_tree[levelput + i] = m_tree[levelget + 2 * i + 1];
        else if (m_less(m_tree[levelget + 2 * i], m_tree[levelget + 2 * i + 1]))
          m_tree[levelput + i] = m_tree[levelget + 2 * i];
        else
          m_tree[levelput + i] = m_tree[levelget + 2 * i + 1];
      }
    }
  }

  int top() const { return m_tree[0]; }

  void replay() {
    int top = m_tree[0];

    int p = (m_tree.size() / 2) + top;

    if (m_less.done(top)) top = -1;  // mark sequence as done

    while (p > 0) {
      m_tree[p] = top;

      p -= (p + 1) % 2;  // round down to left node position

      if (m_tree[p] < 0)
        top = m_tree[p + 1];
      else if (m_tree[p + 1] < 0)
        top = m_tree[p];
      else if (m_less(m_tree[p], m_tree[p + 1]))
        top = m_tree[p];
      else
        top = m_tree[p + 1];

      p /= 2;
    }

    m_tree[p] = top;
  }

  void print() const {
    int levelsize = 1;
    int j = 0;

    for (unsigned int i = 0; i < m_tree.size(); ++i) {
      if (i >= j + levelsize) {
        std::cout << "\n";
        j = i;
        levelsize *= 2;
      }
      std::cout << m_tree[i] << " ";
    }
    std::cout << "\n";
  }
};

template <typename Type, class Comparator>
struct MergeAreasLTCmp {
  std::vector<Type>& v;
  std::vector<int>& pos;
  const int* endpos;
  const Comparator& cmp;

  MergeAreasLTCmp(std::vector<Type>& _v, std::vector<int>& _pos, int* _endpos,
                  const Comparator& _cmp)
      : v(_v), pos(_pos), endpos(_endpos), cmp(_cmp) {}

  bool done(int v) const { return pos[v] >= endpos[v + 1]; }

  bool operator()(int a, int b) const { return cmp(v[pos[a]], v[pos[b]]); }
};

/**
 * Merge an area of ordered sequences as received from other processors:
 *
 * @param v	the complete vector
 * @param area  array of indexes to the adjacent areas first position - size
 * arealen+1 (!)
 * @param arealen number of areas.
 * @param cmp	comparator
 */

template <typename Type, class Comparator>
void merge_areas(std::vector<Type>& v, int* area, int areanum,
                 const Comparator& cmp = std::less<Type>()) {
  std::vector<int> pos(&area[0], &area[areanum + 1]);

  MergeAreasLTCmp<Type, Comparator> ltcmp(v, pos, area, cmp);
  LoserTree<MergeAreasLTCmp<Type, Comparator>> LT(areanum, ltcmp);

  std::vector<Type> out(v.size());

  int top, j = 0;

  while ((top = LT.top()) >= 0) {
    out[j++] = v[pos[top]];

    ++pos[top];

    LT.replay();
  }

  std::swap(v, out);
}

template <typename Type>
void merge_areas(std::vector<Type>& v, int* area, int areanum) {
  return merge_areas(v, area, areanum, std::less<Type>());
}

// **********************************************************************
// * pDCX base class

struct DC3Param {
  static const unsigned int X = 3;
  static const unsigned int D = 2;

  static const unsigned int DC[D];

  static const int cmpDepthRanks[X][X][3];
};

const unsigned int DC3Param::DC[] = {1, 2};

const int DC3Param::cmpDepthRanks[3][3][3] = {
    {{1, 0, 0}, {1, 0, 1}, {2, 1, 1}},
    {{1, 1, 0}, {0, 0, 0}, {0, 0, 0}},
    {{2, 1, 1}, {0, 0, 0}, {0, 0, 0}},
};

struct DC7Param {
  static const unsigned int X = 7;
  static const unsigned int D = 3;

  static const unsigned int DC[D];

  static const int cmpDepthRanks[X][X][3];
};

const unsigned int DC7Param::DC[] = {0, 1, 3};

const int DC7Param::cmpDepthRanks[7][7][3] = {
    {{0, 0, 0},
     {0, 0, 0},
     {1, 1, 0},
     {0, 0, 0},
     {3, 2, 0},
     {3, 2, 1},
     {1, 1, 0}},
    {{0, 0, 0},
     {0, 0, 0},
     {6, 2, 2},
     {0, 0, 0},
     {6, 2, 2},
     {2, 1, 0},
     {2, 1, 1}},
    {{1, 0, 1},
     {6, 2, 2},
     {1, 0, 0},
     {5, 1, 2},
     {6, 2, 2},
     {5, 1, 2},
     {1, 0, 0}},
    {{0, 0, 0},
     {0, 0, 0},
     {5, 2, 1},
     {0, 0, 0},
     {4, 1, 1},
     {5, 2, 2},
     {4, 1, 2}},
    {{3, 0, 2},
     {6, 2, 2},
     {6, 2, 2},
     {4, 1, 1},
     {3, 0, 0},
     {3, 0, 1},
     {4, 1, 2}},
    {{3, 1, 2},
     {2, 0, 1},
     {5, 2, 1},
     {5, 2, 2},
     {3, 1, 0},
     {2, 0, 0},
     {2, 0, 1}},
    {{1, 0, 1},
     {2, 1, 1},
     {1, 0, 0},
     {4, 2, 1},
     {4, 2, 1},
     {2, 1, 0},
     {1, 0, 0}},
};

struct DC13Param {
  static const unsigned int X = 13;
  static const unsigned int D = 4;

  static const unsigned int DC[D];

  static const int cmpDepthRanks[X][X][3];
};

const unsigned int DC13Param::DC[] = {0, 1, 3, 9};

const int DC13Param::cmpDepthRanks[13][13][3] = {
    {{0, 0, 0},
     {0, 0, 0},
     {1, 1, 0},
     {0, 0, 0},
     {9, 3, 1},
     {9, 3, 2},
     {3, 2, 0},
     {9, 3, 3},
     {1, 1, 0},
     {0, 0, 0},
     {3, 2, 0},
     {3, 2, 1},
     {1, 1, 0}},
    {{0, 0, 0},
     {0, 0, 0},
     {12, 3, 3},
     {0, 0, 0},
     {12, 3, 3},
     {8, 2, 1},
     {8, 2, 2},
     {2, 1, 0},
     {8, 2, 3},
     {0, 0, 0},
     {12, 3, 3},
     {2, 1, 0},
     {2, 1, 1}},
    {{1, 0, 1},
     {12, 3, 3},
     {1, 0, 0},
     {11, 2, 3},
     {12, 3, 3},
     {11, 2, 3},
     {7, 1, 1},
     {7, 1, 2},
     {1, 0, 0},
     {7, 1, 3},
     {12, 3, 3},
     {11, 2, 3},
     {1, 0, 0}},
    {{0, 0, 0},
     {0, 0, 0},
     {11, 3, 2},
     {0, 0, 0},
     {10, 2, 2},
     {11, 3, 3},
     {10, 2, 3},
     {6, 1, 1},
     {6, 1, 2},
     {0, 0, 0},
     {6, 1, 2},
     {11, 3, 3},
     {10, 2, 3}},
    {{9, 1, 3},
     {12, 3, 3},
     {12, 3, 3},
     {10, 2, 2},
     {5, 0, 0},
     {9, 1, 2},
     {10, 2, 3},
     {9, 1, 3},
     {5, 0, 1},
     {5, 0, 2},
     {12, 3, 3},
     {5, 0, 2},
     {10, 2, 3}},
    {{9, 2, 3},
     {8, 1, 2},
     {11, 3, 2},
     {11, 3, 3},
     {9, 2, 1},
     {4, 0, 0},
     {8, 1, 2},
     {9, 2, 3},
     {8, 1, 3},
     {4, 0, 1},
     {4, 0, 1},
     {11, 3, 3},
     {4, 0, 2}},
    {{3, 0, 2},
     {8, 2, 2},
     {7, 1, 1},
     {10, 3, 2},
     {10, 3, 2},
     {8, 2, 1},
     {3, 0, 0},
     {7, 1, 2},
     {8, 2, 3},
     {7, 1, 3},
     {3, 0, 0},
     {3, 0, 1},
     {10, 3, 3}},
    {{9, 3, 3},
     {2, 0, 1},
     {7, 2, 1},
     {6, 1, 1},
     {9, 3, 1},
     {9, 3, 2},
     {7, 2, 1},
     {2, 0, 0},
     {6, 1, 2},
     {7, 2, 3},
     {6, 1, 2},
     {2, 0, 0},
     {2, 0, 1}},
    {{1, 0, 1},
     {8, 3, 2},
     {1, 0, 0},
     {6, 2, 1},
     {5, 1, 0},
     {8, 3, 1},
     {8, 3, 2},
     {6, 2, 1},
     {1, 0, 0},
     {5, 1, 2},
     {6, 2, 2},
     {5, 1, 2},
     {1, 0, 0}},
    {{0, 0, 0},
     {0, 0, 0},
     {7, 3, 1},
     {0, 0, 0},
     {5, 2, 0},
     {4, 1, 0},
     {7, 3, 1},
     {7, 3, 2},
     {5, 2, 1},
     {0, 0, 0},
     {4, 1, 1},
     {5, 2, 2},
     {4, 1, 2}},
    {{3, 0, 2},
     {12, 3, 3},
     {12, 3, 3},
     {6, 2, 1},
     {12, 3, 3},
     {4, 1, 0},
     {3, 0, 0},
     {6, 2, 1},
     {6, 2, 2},
     {4, 1, 1},
     {3, 0, 0},
     {3, 0, 1},
     {4, 1, 2}},
    {{3, 1, 2},
     {2, 0, 1},
     {11, 3, 2},
     {11, 3, 3},
     {5, 2, 0},
     {11, 3, 3},
     {3, 1, 0},
     {2, 0, 0},
     {5, 2, 1},
     {5, 2, 2},
     {3, 1, 0},
     {2, 0, 0},
     {2, 0, 1}},
    {{1, 0, 1},
     {2, 1, 1},
     {1, 0, 0},
     {10, 3, 2},
     {10, 3, 2},
     {4, 2, 0},
     {10, 3, 3},
     {2, 1, 0},
     {1, 0, 0},
     {4, 2, 1},
     {4, 2, 1},
     {2, 1, 0},
     {1, 0, 0}},
};

template <typename Type>
std::string strC(const Type& t) {
  std::ostringstream os;
  os << t;
  return os.str();
}

template <>
std::string strC(const char& c) {
  std::ostringstream os;
  if (isprint(c) && !isspace(c))
    os << (char)c;
  else
    os << (int)c;
  return os.str();
}

template <>
std::string strC(const unsigned char& c) {
  std::ostringstream os;
  if (isprint(c) && !isspace(c))
    os << (char)c;
  else
    os << (int)c;
  return os.str();
}

template <typename Type>
void vector_free(std::vector<Type>& v) {
  std::vector<Type> v2;
  std::swap(v, v2);
}

template <typename Type>
void exclusive_prefixsum(Type* array, unsigned int size) {
  uint sum = 0;
  for (unsigned int i = 0; i < size; ++i) {
    uint newsum = sum + array[i];
    array[i] = sum;
    sum = newsum;
  }
  array[size] = sum;
}

