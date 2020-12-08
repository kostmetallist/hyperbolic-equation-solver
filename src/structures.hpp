#ifndef STRUCTURES_HPP
#define STRUCTURES_HPP


typedef struct {
    int from;
    int to;
} Range;

typedef struct {
    Range x;
    Range y;
    Range z;

    Range const& operator[](int index) const {
        if (not index) {
            return x;
        } else if (index == 1) {
            return y;
        } else {
            return z;
        }
    }

} ProcessBlockArea;

typedef struct {
    ProcessBlockArea inner;
    // extended area represents the union of inner cells and interface ones
    ProcessBlockArea extended;
} ProcessBlock;

typedef struct {
    int lowest;
    int highest;
} AdjacentDirections;

class MatrixAccessor {
private:
    double *_data;
    int _xn, _yn, _zn;
    int _offset_x, _offset_y, _offset_z;

public:

    MatrixAccessor(double *data, int xn, int yn, int zn) {
        _data = data;
        _xn = xn;
        _yn = yn;
        _zn = zn;
        _offset_x = 0;
        _offset_y = 0;
        _offset_z = 0;
    }

    int derive_index(int x, int y, int z, bool use_offset = false) {
        if (use_offset) {
            return (x + _offset_x) * _zn * _yn +
                   (y + _offset_y) * _zn +
                   (z + _offset_z);
        } else {
            return x * _zn * _yn + y * _zn + z;
        }
    }

    void configure_offsets(int offset_x, int offset_y, int offset_z) {
        _offset_x = offset_x;
        _offset_y = offset_y;
        _offset_z = offset_z;
    }

    double get(int x, int y, int z, bool use_offset = false) {
        if (use_offset) {
            return _data[(x + _offset_x) * _zn * _yn +
                         (y + _offset_y) * _zn +
                         (z + _offset_z)];
        } else {
            return _data[x * _zn * _yn + y * _zn + z];
        }
    }

    void set(int x, int y, int z, double value, bool use_offset = false) {
        if (use_offset) {
            _data[(x + _offset_x) * _zn * _yn +
                  (y + _offset_y) * _zn +
                  (z + _offset_z)] = value;
        } else {
            _data[x * _zn * _yn + y * _zn + z] = value;
        }
    }
};

#endif
