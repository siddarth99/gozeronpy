# gozeronpy

[![CI](https://github.com/siddarth99/gozeronpy/actions/workflows/ci.yml/badge.svg)](https://github.com/siddarth99/gozeronpy/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/siddarth99/gozeronpy)](https://goreportcard.com/report/github.com/siddarth99/gozeronpy)
[![Go Reference](https://pkg.go.dev/badge/github.com/siddarth99/gozeronpy.svg)](https://pkg.go.dev/github.com/siddarth99/gozeronpy)

A fast, zero-copy Go library for reading and writing NumPy `.npy` files.

The array's **shape and element type are detected automatically** from the
file — just like `numpy.load` — and on the common little-endian case the raw
file bytes are reinterpreted directly as a Go slice with **no per-element
copy**.

```go
arr, _ := npy.Open("data.npy")
fmt.Println(arr.Shape)   // e.g. [2 3] — discovered from the file
xs, _ := arr.Float64()   // []float64 view of the data
```

## Why it's fast

- **Zero-copy reads.** When the file byte order matches the host (little-endian
  on amd64/arm64), the data region is reinterpreted as a Go slice via `unsafe`
  rather than parsed element by element. Reads run at memory bandwidth
  (~12 GB/s on an M-series laptop).
- **RAM-aware access modes.** `Auto` (default) compares the data size to
  available system memory: small files are read into memory; files large
  relative to RAM are memory-mapped so you can work with arrays bigger than
  RAM without loading them.
- **Single aligned allocation.** The in-memory path allocates one 8-byte
  aligned buffer for the whole array; the mmap path allocates almost nothing.

```
BenchmarkLoadInMemory-10    12360 MB/s     17 allocs/op   (64 MiB float64 array)
BenchmarkLoadMmap-10         9380 MB/s     16 allocs/op   (incl. page faults)
BenchmarkOpenDynamic-10     12319 MB/s     15 allocs/op
BenchmarkSave-10             3275 MB/s     17 allocs/op
```

## Install

```sh
go get github.com/siddarth99/gozeronpy
```

```go
import npy "github.com/siddarth99/gozeronpy"
```

## Reading

### Dynamic (NumPy-like)

The element type is whatever the file contains; `Data` holds the matching Go
slice (`[]float64`, `[]int32`, …).

```go
arr, err := npy.Open("data.npy")
if err != nil { /* ... */ }
defer arr.Close() // releases the mapping if the array was memory-mapped

fmt.Println(arr.Shape)          // []int, auto-detected
fmt.Println(arr.Dtype)          // e.g. <f8
fmt.Println(arr.Dtype.GoType()) // "float64"

switch v := arr.Data.(type) {
case []float64:
    _ = v
case []int32:
    _ = v
}

// Or pull a typed view directly (errors if the dtype doesn't match):
xs, err := arr.Float64()
ys, err := npy.Values[int32](arr)

// Or convert any numeric dtype to float64 (copying):
f, err := arr.AsFloat64()
```

### Typed (fastest, no runtime assertions)

If you know the element type at compile time, ask for it directly:

```go
nd, err := npy.Load[float64]("data.npy")
if err != nil { /* ... */ }
defer nd.Close()

_ = nd.Values // []float64
_ = nd.Shape  // []int
```

`Load` returns an error if the file's element kind/size doesn't match `T`
(byte order is converted automatically).

### From a stream

```go
arr, err := npy.Decode(r)        // dynamic, reads into memory
nd,  err := npy.Read[float64](r) // typed, reads into memory
```

## Writing

Shape is optional — pass `nil` for a 1-D array, or give an explicit shape whose
product matches the data length.

```go
npy.Save("out.npy", []float64{1, 2, 3, 4}, []int{2, 2})
npy.Save("vec.npy", []int32{1, 2, 3}, nil)                  // 1-D
npy.Save("col.npy", data, []int{2, 2}, npy.WithFortran(true))

// To a stream:
npy.Write(w, []float64{1, 2, 3, 4}, []int{2, 2})
```

## Archives (`.npz`)

A `.npz` file is a ZIP archive of `.npy` entries. Array names are the entry
names without the `.npy` suffix, matching `numpy.load(...).files`.

### Reading

```go
arc, err := npy.OpenZip("data.npz")
if err != nil { /* ... */ }
defer arc.Close()

fmt.Println(arc.Names())          // []string in storage order

arr, _ := arc.Array("x")          // dynamic
nd,  _ := npy.ZipValues[float64](arc, "x") // typed
all, _ := arc.All()               // map[string]*npy.Array

// From an io.ReaderAt instead of a path:
arc, _ = npy.ReadZip(readerAt, size)
```

Like `Open`, `OpenZip` can memory-map the archive (`WithMode`): uncompressed
entries are then read zero-copy directly from the mapping, and compressed
entries are inflated into memory.

### Writing

```go
// All at once (uncompressed, like numpy.savez):
npy.SaveZip("out.npz", []npy.NamedArray{
    {Name: "x", Data: []float64{1, 2, 3, 4}, Shape: []int{2, 2}},
    {Name: "y", Data: []int32{5, 6, 7}},                  // nil shape => 1-D
})

// Compressed (like numpy.savez_compressed):
npy.SaveZip("out.npz", arrays, npy.WithCompression(true))

// Incrementally, without holding every array in memory at once:
zw, _ := npy.CreateZip("out.npz")
npy.AddTyped(zw, "a", []float64{1, 2, 3}, nil)
zw.Add("b", []int32{4, 5, 6}, []int{3})
zw.Close()
```

## Access modes

```go
npy.Open("big.npy", npy.WithMode(npy.Auto))               // default
npy.Open("big.npy", npy.WithMode(npy.InMemory))           // always read into RAM
npy.Load[float64]("big.npy", npy.WithMode(npy.Mmap))      // memory-map
npy.Open("big.npy", npy.WithMaxRAMFraction(0.25))         // Auto threshold
```

- **`Auto`** — memory-map when the file is larger than
  `MaxRAMFraction × system memory` (default 0.5); otherwise read into memory.
  When system memory can't be determined, falls back to a 512 MiB threshold.
- **`InMemory`** — always read the whole array into a heap buffer.
- **`Mmap`** — memory-map the file (transparently falls back to `InMemory` on
  platforms without mmap, e.g. Windows).

**Lifetime:** for a memory-mapped array, `Data`/`Values` alias the mapping, so
call `Close()` when you're done. A finalizer unmaps as a safety net, but
explicit `Close` is preferred. In-memory arrays own their data; `Close` is a
no-op.

## Supported dtypes

| NumPy    | Go            |   | NumPy     | Go                   |
|----------|---------------|---|-----------|----------------------|
| f4 / f8  | float32/64    |   | u1..u8    | uint8..64            |
| i1..i8   | int8..64      |   | b1        | bool                 |
| c8 / c16 | complex64/128 |   | (le & be) | byte-swapped on read |

Both little- and big-endian files are read correctly (big-endian is converted
to host order). Fortran (column-major) order is detected and exposed via the
`Fortran` flag; the data is returned in its on-disk storage order. Structured /
record dtypes, float16, datetime and object arrays are not supported.

## Correctness

The library is cross-validated against real NumPy: a test generates `.npy`
files with NumPy across every dtype, both byte orders, C/Fortran order and
0-D…3-D shapes (122 cases), reads them in Go, and checks shapes, dtypes and
values match — and writes files in Go that NumPy then reads back. The same is
done for `.npz` archives, both uncompressed and compressed.

```sh
go test ./...              # pure-Go tests; NumPy tests run if python3+numpy present
go test -bench=. ./...     # benchmarks
```

## License

GPL-3.0 (see [LICENSE](LICENSE)).
