"""Minimal read-only NetCDF4/HDF5 reader via ctypes on the system libhdf5 (no h5py/xarray here)."""
import ctypes as C
import numpy as np

def _load_hdf5():
    import ctypes.util, os
    names = [os.environ.get("HDF5_LIB", ""), "libhdf5_serial.so.103", "libhdf5.so.103", "libhdf5.so.200", "libhdf5.so.310",
             "libhdf5.so", "libhdf5.dylib", "hdf5.dll", ctypes.util.find_library("hdf5_serial") or "", ctypes.util.find_library("hdf5") or ""]
    for n in names:
        if not n:
            continue
        try:
            return C.CDLL(n)
        except OSError:
            pass
    raise OSError("HDF5 shared library not found; install libhdf5 or set HDF5_LIB to its path")
_lib = _load_hdf5()
hid_t = C.c_int64
herr_t = C.c_int
hsize_t = C.c_ulonglong
_lib.H5open()

def _g(name):
    return hid_t.in_dll(_lib, name).value

NATIVE = {
    ('f', 4): _g("H5T_NATIVE_FLOAT_g"), ('f', 8): _g("H5T_NATIVE_DOUBLE_g"),
    ('i', 1): _g("H5T_NATIVE_SCHAR_g"), ('i', 2): _g("H5T_NATIVE_SHORT_g"),
    ('i', 4): _g("H5T_NATIVE_INT_g"), ('i', 8): _g("H5T_NATIVE_LLONG_g"),
    ('u', 1): _g("H5T_NATIVE_UCHAR_g"), ('u', 2): _g("H5T_NATIVE_USHORT_g"),
    ('u', 4): _g("H5T_NATIVE_UINT_g"), ('u', 8): _g("H5T_NATIVE_ULLONG_g"),
}
for fn, res, args in [
    ("H5Fopen", hid_t, [C.c_char_p, C.c_uint, hid_t]), ("H5Fclose", herr_t, [hid_t]),
    ("H5Dopen2", hid_t, [hid_t, C.c_char_p, hid_t]), ("H5Dclose", herr_t, [hid_t]),
    ("H5Dget_space", hid_t, [hid_t]), ("H5Dget_type", hid_t, [hid_t]),
    ("H5Sget_simple_extent_ndims", C.c_int, [hid_t]),
    ("H5Sget_simple_extent_dims", C.c_int, [hid_t, C.POINTER(hsize_t), C.POINTER(hsize_t)]),
    ("H5Sclose", herr_t, [hid_t]), ("H5Tclose", herr_t, [hid_t]),
    ("H5Tget_class", C.c_int, [hid_t]), ("H5Tget_size", C.c_size_t, [hid_t]),
    ("H5Tget_sign", C.c_int, [hid_t]), ("H5Tis_variable_str", C.c_int, [hid_t]),
    ("H5Dread", herr_t, [hid_t, hid_t, hid_t, hid_t, hid_t, C.c_void_p]),
    ("H5Gopen2", hid_t, [hid_t, C.c_char_p, hid_t]), ("H5Gclose", herr_t, [hid_t]),
    ("H5Lget_name_by_idx", C.c_ssize_t, [hid_t, C.c_char_p, C.c_int, C.c_int, hsize_t, C.c_char_p, C.c_size_t, hid_t]),
    ("H5Oget_info_by_name", herr_t, [hid_t, C.c_char_p, C.c_void_p, hid_t]),
    ("H5Aget_num_attrs", C.c_int, [hid_t]),
    ("H5Aopen_by_idx", hid_t, [hid_t, C.c_char_p, C.c_int, C.c_int, hsize_t, hid_t, hid_t]),
    ("H5Aget_name", C.c_ssize_t, [hid_t, C.c_size_t, C.c_char_p]),
    ("H5Aget_type", hid_t, [hid_t]), ("H5Aget_space", hid_t, [hid_t]),
    ("H5Aread", herr_t, [hid_t, hid_t, C.c_void_p]), ("H5Aclose", herr_t, [hid_t]),
    ("H5Sselect_hyperslab", herr_t, [hid_t, C.c_int, C.POINTER(hsize_t), C.POINTER(hsize_t), C.POINTER(hsize_t), C.POINTER(hsize_t)]),
    ("H5Screate_simple", hid_t, [C.c_int, C.POINTER(hsize_t), C.POINTER(hsize_t)]),
    ("H5Eset_auto2", herr_t, [hid_t, C.c_void_p, C.c_void_p]),
]:
    f = getattr(_lib, fn); f.restype = res; f.argtypes = args
_lib.H5Eset_auto2(0, None, None)  # silence HDF5 error stack printing

class H5Info(C.Structure):  # H5O_info_t (1.10) - we only need 'type' field offset
    _fields_ = [("fileno", C.c_ulong), ("addr", C.c_uint64), ("type", C.c_int)]

def _dtype_of(t):
    cls = _lib.H5Tget_class(t); size = _lib.H5Tget_size(t)
    if cls == 1: return ('f', size)
    if cls == 0: return ('i' if _lib.H5Tget_sign(t) == 1 else 'u', size)
    if cls == 3: return ('s', size)
    return (None, size)

def _shape(space):
    n = _lib.H5Sget_simple_extent_ndims(space)
    if n <= 0: return ()
    dims = (hsize_t * n)(); _lib.H5Sget_simple_extent_dims(space, dims, None)
    return tuple(int(d) for d in dims)

class File:
    def __init__(self, path):
        self.path = path
        self.fid = _lib.H5Fopen(path.encode(), 0, 0)
        if self.fid < 0: raise IOError(path)
    def close(self): _lib.H5Fclose(self.fid)
    def names(self, group="/"):
        g = _lib.H5Gopen2(self.fid, group.encode(), 0); out = []; i = 0
        while True:
            buf = C.create_string_buffer(512)
            r = _lib.H5Lget_name_by_idx(g, b".", 0, 0, i, buf, 512, 0)
            if r < 0: break
            out.append(buf.value.decode()); i += 1
        _lib.H5Gclose(g); return out
    def attrs(self, name):
        obj = _lib.H5Dopen2(self.fid, name.encode(), 0) if name != "/" else _lib.H5Gopen2(self.fid, b"/", 0)
        out = {}
        for i in range(_lib.H5Aget_num_attrs(obj)):
            a = _lib.H5Aopen_by_idx(obj, b".", 0, 0, i, 0, 0)
            nb = C.create_string_buffer(256); _lib.H5Aget_name(a, 256, nb)
            t = _lib.H5Aget_type(a); sp = _lib.H5Aget_space(a); shp = _shape(sp)
            kind, size = _dtype_of(t)
            n = int(np.prod(shp)) if shp else 1
            try:
                if kind == 's':
                    if _lib.H5Tis_variable_str(t) > 0:
                        ptrs = (C.c_char_p * n)(); _lib.H5Aread(a, t, ptrs)
                        val = [p.decode(errors='replace') if p else '' for p in ptrs]
                        val = val[0] if n == 1 else val
                    else:
                        buf = C.create_string_buffer(size * n); _lib.H5Aread(a, t, buf)
                        val = buf.raw.rstrip(b"\0").decode(errors='replace')
                elif (kind, size) in NATIVE:
                    arr = np.empty(n, dtype=f"{kind}{size}")
                    _lib.H5Aread(a, NATIVE[(kind, size)], arr.ctypes.data)
                    val = arr[0] if n == 1 else arr
                else: val = None
            except Exception as e:
                val = f"<unreadable {e}>"
            out[nb.value.decode()] = val
            _lib.H5Sclose(sp); _lib.H5Tclose(t); _lib.H5Aclose(a)
        if name != "/": _lib.H5Dclose(obj)
        else: _lib.H5Gclose(obj)
        return out
    def info(self, name):
        d = _lib.H5Dopen2(self.fid, name.encode(), 0)
        if d < 0: return None
        sp = _lib.H5Dget_space(d); t = _lib.H5Dget_type(d)
        r = (_shape(sp), _dtype_of(t))
        _lib.H5Sclose(sp); _lib.H5Tclose(t); _lib.H5Dclose(d); return r
    def read(self, name, sel=None, raw=False):
        """sel: tuple of (start, count) per dim, or None for full."""
        d = _lib.H5Dopen2(self.fid, name.encode(), 0)
        sp = _lib.H5Dget_space(d); t = _lib.H5Dget_type(d)
        shp = _shape(sp); kind, size = _dtype_of(t)
        dt = np.dtype(f"S{size}") if kind == 's' else np.dtype(f"{kind}{size}")
        mt = t if kind == 's' else NATIVE[(kind, size)]
        if sel is None:
            arr = np.empty(shp, dtype=dt)
            err = _lib.H5Dread(d, mt, 0, 0, 0, arr.ctypes.data)
        else:
            nd = len(shp)
            start = (hsize_t * nd)(*[s for s, c in sel]); count = (hsize_t * nd)(*[c for s, c in sel])
            _lib.H5Sselect_hyperslab(sp, 0, start, None, count, None)
            mem = _lib.H5Screate_simple(nd, count, None)
            arr = np.empty(tuple(c for s, c in sel), dtype=dt)
            err = _lib.H5Dread(d, mt, mem, sp, 0, arr.ctypes.data)
            _lib.H5Sclose(mem)
        _lib.H5Sclose(sp); _lib.H5Tclose(t); _lib.H5Dclose(d)
        if err < 0: raise IOError(f"read failed {name}")
        if raw or kind == 's': return arr
        a = self.attrs(name)
        out = arr.astype("f8") if kind in 'iu' or 'scale_factor' in a else arr
        fv = a.get('_FillValue', None)
        if fv is not None and np.ndim(fv) == 0:
            out = np.where(arr == fv, np.nan, out.astype('f8') if out.dtype.kind != 'f' else out)
        if 'scale_factor' in a: out = out * float(a['scale_factor'])
        if 'add_offset' in a: out = out + float(a['add_offset'])
        return out
