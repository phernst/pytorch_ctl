#ifndef ROWITERATOR_H
#define ROWITERATOR_H

#include <iterator>

// assumes row-major order (=todo?)
template<typename T, typename ssize_t=size_t>
class RowIterator {
public:
    class iterator {
    public:
        using difference_type = long;
        using value_type = T;
        using pointer = const T*;
        using reference = const T&;
        using iterator_category = std::forward_iterator_tag;

        iterator(pointer buffer = nullptr, ssize_t nrows = 0, ssize_t ncols = 0, ssize_t row = 0, ssize_t col = 0)
            : _buffer{buffer}, _nrows{nrows}, _ncols{ncols}, _row{row}, _col{col} {}
        iterator& operator++() {
            _row = (_row + 1)%_nrows;
            _col += _row == 0 ? 1 : 0;
            return *this;
        }
        iterator operator++(int) {
            iterator ret { *this };
            ++(*this);
            return ret;
        }
        bool operator==(iterator other) const {
            return _buffer == other._buffer &&
                    _nrows == other._nrows &&
                    _ncols == other._ncols &&
                    _row == other._row &&
                    _col == other._col;
        }
        bool operator!=(iterator other) const { return !(*this == other); }
        value_type operator*() { return _buffer[_row*_ncols + _col]; }
    private:
        pointer _buffer;
        ssize_t _nrows;
        ssize_t _ncols;
        ssize_t _row;
        ssize_t _col;
    };

    // buffer of size nrows*ncols, iterator starting at position (row,col)
    RowIterator(T* buffer, ssize_t nrows, ssize_t ncols)
        : _buffer{buffer}, _nrows{nrows}, _ncols{ncols} {}

    iterator begin(ssize_t col=0) { return {_buffer, _nrows, _ncols, 0, col}; }
    iterator end(ssize_t col=-1) { return {_buffer, _nrows, _ncols, 0, col > -1 ? col : _ncols}; }

private:
    T* _buffer;
    ssize_t _nrows;
    ssize_t _ncols;
};

#endif