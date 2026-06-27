package npy

import "fmt"

// Array is the dynamically-typed result of reading a .npy file. The element
// type is discovered from the file and stored in Data as the matching Go
// slice type (for example []float64 or []int32).
type Array struct {
	Shape   []int // dimensions, as in NumPy
	Fortran bool  // true if the data is in Fortran (column-major) order
	Dtype   DType // the on-disk element type
	Data    any   // the typed element slice, e.g. []float64
	closer  func() error
}

// Len returns the total number of elements (the product of Shape).
func (a *Array) Len() int { return numElements(a.Shape) }

// Close releases resources associated with the array. It is only meaningful
// for memory-mapped arrays; for in-memory arrays it is a no-op. After Close
// the Data slice of a memory-mapped array must not be used.
func (a *Array) Close() error {
	if a.closer != nil {
		err := a.closer()
		a.closer = nil
		return err
	}
	return nil
}

// Values returns the array's data as a typed slice. It returns an error if T
// does not match the array's on-disk dtype.
func Values[T Element](a *Array) ([]T, error) {
	if s, ok := a.Data.([]T); ok {
		return s, nil
	}
	var z T
	return nil, fmt.Errorf("npy: array dtype is %s (%s), not %T", a.Dtype, a.Dtype.GoType(), z)
}

// The following convenience accessors return the data slice when the dtype
// matches and an error otherwise.

func (a *Array) Float64() ([]float64, error)       { return Values[float64](a) }
func (a *Array) Float32() ([]float32, error)       { return Values[float32](a) }
func (a *Array) Int64() ([]int64, error)           { return Values[int64](a) }
func (a *Array) Int32() ([]int32, error)           { return Values[int32](a) }
func (a *Array) Int16() ([]int16, error)           { return Values[int16](a) }
func (a *Array) Int8() ([]int8, error)             { return Values[int8](a) }
func (a *Array) Uint64() ([]uint64, error)         { return Values[uint64](a) }
func (a *Array) Uint32() ([]uint32, error)         { return Values[uint32](a) }
func (a *Array) Uint16() ([]uint16, error)         { return Values[uint16](a) }
func (a *Array) Uint8() ([]uint8, error)           { return Values[uint8](a) }
func (a *Array) Bool() ([]bool, error)             { return Values[bool](a) }
func (a *Array) Complex64() ([]complex64, error)   { return Values[complex64](a) }
func (a *Array) Complex128() ([]complex128, error) { return Values[complex128](a) }

// AsFloat64 returns a copy of the data converted to float64, regardless of the
// underlying numeric dtype. Boolean and complex arrays are not supported.
func (a *Array) AsFloat64() ([]float64, error) {
	switch s := a.Data.(type) {
	case []float64:
		out := make([]float64, len(s))
		copy(out, s)
		return out, nil
	case []float32:
		return convertTo[float64](s), nil
	case []int8:
		return convertTo[float64](s), nil
	case []int16:
		return convertTo[float64](s), nil
	case []int32:
		return convertTo[float64](s), nil
	case []int64:
		return convertTo[float64](s), nil
	case []uint8:
		return convertTo[float64](s), nil
	case []uint16:
		return convertTo[float64](s), nil
	case []uint32:
		return convertTo[float64](s), nil
	case []uint64:
		return convertTo[float64](s), nil
	default:
		return nil, fmt.Errorf("npy: cannot convert dtype %s to float64", a.Dtype)
	}
}

// realNumber is the set of element types convertible to a Go float64.
type realNumber interface {
	~int8 | ~int16 | ~int32 | ~int64 |
		~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}

func convertTo[D, S realNumber](in []S) []D {
	out := make([]D, len(in))
	for i, v := range in {
		out[i] = D(v)
	}
	return out
}

// NDArray is the statically-typed result of reading a .npy file with a known
// element type. It is the fastest API: there are no runtime type assertions.
type NDArray[T Element] struct {
	Values  []T   // the element data
	Shape   []int // dimensions
	Fortran bool  // true if the data is in Fortran (column-major) order
	closer  func() error
}

// Len returns the total number of elements.
func (n *NDArray[T]) Len() int { return len(n.Values) }

// Close releases resources for memory-mapped arrays; otherwise a no-op. After
// Close the Values slice of a memory-mapped array must not be used.
func (n *NDArray[T]) Close() error {
	if n.closer != nil {
		err := n.closer()
		n.closer = nil
		return err
	}
	return nil
}
