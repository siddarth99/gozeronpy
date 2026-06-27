// Package npy reads and writes NumPy .npy files.
//
// It is designed to be fast: when the on-disk byte order matches the host
// (the common little-endian case) the raw file bytes are reinterpreted
// directly as a Go slice with no per-element copy. For files that are large
// relative to available RAM it can memory-map the data instead of loading it.
//
// There are two API tiers:
//
//   - Dynamic, Python-like:
//
//     arr, err := npy.Open("data.npy")
//     fmt.Println(arr.Shape)            // shape auto-detected
//     xs, _ := arr.Float64()            // typed view of the data
//
//   - Generic, fastest (no runtime type assertions):
//
//     nd, err := npy.Load[float64]("data.npy")
//     _ = nd.Values                     // []float64
//     _ = nd.Shape                      // []int
//
// Writing mirrors this:
//
//	npy.Save("out.npy", []float64{1, 2, 3, 4}, []int{2, 2})
//	npy.Write(w, []int32{1, 2, 3}, nil) // nil shape => 1-D
//
// The .npy format is documented at
// https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
package npy

import (
	"encoding/binary"
	"unsafe"
)

// Element is the set of array element types this package can read and write.
// It mirrors the numeric (and boolean) dtypes NumPy stores in .npy files.
type Element interface {
	~int8 | ~int16 | ~int32 | ~int64 |
		~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64 |
		~complex64 | ~complex128 |
		~bool
}

// hostOrder is the byte order of the machine running this code.
var hostOrder = func() binary.ByteOrder {
	var i uint16 = 1
	if *(*byte)(unsafe.Pointer(&i)) == 1 {
		return binary.LittleEndian
	}
	return binary.BigEndian
}()

// hostIsLittle reports whether the host is little-endian.
var hostIsLittle = hostOrder == binary.LittleEndian

// bytesAsSlice reinterprets a byte slice as a slice of T without copying.
// len(b) must be a multiple of the size of T and &b[0] must be suitably
// aligned for T (guaranteed for freshly allocated buffers and for the
// 64-byte-aligned data region of a .npy file).
func bytesAsSlice[T any](b []byte) []T {
	if len(b) == 0 {
		return nil
	}
	var z T
	n := len(b) / int(unsafe.Sizeof(z))
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*T)(unsafe.Pointer(unsafe.SliceData(b))), n)
}

// sliceAsBytes reinterprets a slice of T as a byte slice without copying.
func sliceAsBytes[T any](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	var z T
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), len(s)*int(unsafe.Sizeof(z)))
}

// numElements returns the product of the dimensions. The empty shape (a
// 0-dimensional array / NumPy scalar) has exactly one element.
func numElements(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}
