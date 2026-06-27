package npy

import (
	"bufio"
	"fmt"
	"io"
	"os"
)

// Save writes a typed slice to a .npy file. data must be one of the supported
// element slice types (for example []float64 or []int32). If shape is nil the
// array is written as 1-D with length len(data); otherwise the product of
// shape must equal len(data).
//
//	npy.Save("out.npy", []float64{1, 2, 3, 4}, []int{2, 2})
func Save(path string, data any, shape []int, opts ...Option) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	bw := bufio.NewWriter(f)
	if err := encode(bw, data, shape, resolve(opts)); err != nil {
		f.Close()
		return err
	}
	if err := bw.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Write writes a statically-typed slice to a .npy stream. A nil shape means a
// 1-D array of length len(data).
func Write[T Element](w io.Writer, data []T, shape []int, opts ...Option) error {
	cfg := resolve(opts)
	dt, err := dtypeFor[T]()
	if err != nil {
		return err
	}
	shape = normalizeShape(shape, len(data))
	if numElements(shape) != len(data) {
		return fmt.Errorf("npy: shape %v implies %d elements but data has %d", shape, numElements(shape), len(data))
	}
	if err := writeHeader(w, dt, cfg.fortran, shape); err != nil {
		return err
	}
	if len(data) == 0 {
		return nil
	}
	_, err = w.Write(sliceAsBytes(data))
	return err
}

// encode writes the dynamically-typed data to w.
func encode(w io.Writer, data any, shape []int, cfg config) error {
	dt, raw, n, err := rawBytes(data)
	if err != nil {
		return err
	}
	shape = normalizeShape(shape, n)
	if numElements(shape) != n {
		return fmt.Errorf("npy: shape %v implies %d elements but data has %d", shape, numElements(shape), n)
	}
	if err := writeHeader(w, dt, cfg.fortran, shape); err != nil {
		return err
	}
	if n == 0 {
		return nil
	}
	_, err = w.Write(raw)
	return err
}

// normalizeShape returns shape, defaulting to a 1-D shape of length n when nil.
func normalizeShape(shape []int, n int) []int {
	if shape == nil {
		return []int{n}
	}
	return shape
}

// rawBytes returns the dtype, host-order byte view and element count for a
// supported slice value.
func rawBytes(data any) (DType, []byte, int, error) {
	switch s := data.(type) {
	case []float32:
		return elemBytes(s)
	case []float64:
		return elemBytes(s)
	case []int8:
		return elemBytes(s)
	case []int16:
		return elemBytes(s)
	case []int32:
		return elemBytes(s)
	case []int64:
		return elemBytes(s)
	case []uint8:
		return elemBytes(s)
	case []uint16:
		return elemBytes(s)
	case []uint32:
		return elemBytes(s)
	case []uint64:
		return elemBytes(s)
	case []bool:
		return elemBytes(s)
	case []complex64:
		return elemBytes(s)
	case []complex128:
		return elemBytes(s)
	default:
		return DType{}, nil, 0, fmt.Errorf("npy: unsupported data type %T (want a slice of a numeric, bool or complex type)", data)
	}
}

// elemBytes builds the dtype and byte view for a typed slice.
func elemBytes[T Element](s []T) (DType, []byte, int, error) {
	dt, err := dtypeFor[T]()
	if err != nil {
		return DType{}, nil, 0, err
	}
	return dt, sliceAsBytes(s), len(s), nil
}
