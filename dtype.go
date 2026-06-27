package npy

import (
	"encoding/binary"
	"fmt"
	"reflect"
	"strconv"
)

// DType describes the element type of an array, mirroring a NumPy dtype.
type DType struct {
	Kind     byte             // 'f' float, 'i' int, 'u' uint, 'b' bool, 'c' complex
	ItemSize int              // bytes per element
	Order    binary.ByteOrder // on-disk byte order; nil for single-byte types
	Descr    string           // the original descr string, e.g. "<f8"
}

// key uniquely identifies a supported (kind, size) pair.
func (d DType) key() int { return int(d.Kind)<<8 | d.ItemSize }

// needSwap reports whether the on-disk bytes must be byte-swapped to match the
// host before they can be reinterpreted as native values.
func (d DType) needSwap() bool {
	if d.ItemSize == 1 || d.Order == nil {
		return false
	}
	return d.Order != hostOrder
}

// String returns the NumPy-style descr, e.g. "<f8".
func (d DType) String() string { return d.Descr }

// GoType returns the name of the Go type used to represent this dtype.
func (d DType) GoType() string {
	if t, ok := goTypeNames[d.key()]; ok {
		return t
	}
	return "unsupported"
}

var goTypeNames = map[int]string{
	key('f', 4): "float32", key('f', 8): "float64",
	key('i', 1): "int8", key('i', 2): "int16", key('i', 4): "int32", key('i', 8): "int64",
	key('u', 1): "uint8", key('u', 2): "uint16", key('u', 4): "uint32", key('u', 8): "uint64",
	key('b', 1): "bool",
	key('c', 8): "complex64", key('c', 16): "complex128",
}

func key(kind byte, size int) int { return int(kind)<<8 | size }

// parseDescr parses a NumPy descr string such as "<f8", "|u1" or ">i4".
func parseDescr(s string) (DType, error) {
	if len(s) < 2 {
		return DType{}, fmt.Errorf("npy: invalid dtype descr %q", s)
	}
	var order binary.ByteOrder
	rest := s
	switch s[0] {
	case '<':
		order = binary.LittleEndian
		rest = s[1:]
	case '>':
		order = binary.BigEndian
		rest = s[1:]
	case '=':
		order = hostOrder
		rest = s[1:]
	case '|':
		order = nil // not applicable (single-byte type)
		rest = s[1:]
	default:
		order = hostOrder // no byte-order char: assume native
	}
	if len(rest) < 2 {
		return DType{}, fmt.Errorf("npy: invalid dtype descr %q", s)
	}
	kind := rest[0]
	itemSize, err := strconv.Atoi(rest[1:])
	if err != nil {
		return DType{}, fmt.Errorf("npy: invalid dtype descr %q: %w", s, err)
	}
	dt := DType{Kind: kind, ItemSize: itemSize, Order: order, Descr: s}
	if _, ok := goTypeNames[dt.key()]; !ok {
		return DType{}, fmt.Errorf("npy: unsupported dtype %q (Go has no native %c%d type)", s, kind, itemSize)
	}
	if itemSize == 1 {
		dt.Order = nil // byte order is meaningless for single-byte elements
	}
	return dt, nil
}

// dtypeFor returns the DType used to store values of type T on disk, in the
// host byte order.
func dtypeFor[T Element]() (DType, error) {
	var z T
	var kind byte
	var size int
	switch reflect.TypeOf(z).Kind() {
	case reflect.Float32:
		kind, size = 'f', 4
	case reflect.Float64:
		kind, size = 'f', 8
	case reflect.Int8:
		kind, size = 'i', 1
	case reflect.Int16:
		kind, size = 'i', 2
	case reflect.Int32:
		kind, size = 'i', 4
	case reflect.Int64:
		kind, size = 'i', 8
	case reflect.Uint8:
		kind, size = 'u', 1
	case reflect.Uint16:
		kind, size = 'u', 2
	case reflect.Uint32:
		kind, size = 'u', 4
	case reflect.Uint64:
		kind, size = 'u', 8
	case reflect.Bool:
		kind, size = 'b', 1
	case reflect.Complex64:
		kind, size = 'c', 8
	case reflect.Complex128:
		kind, size = 'c', 16
	default:
		return DType{}, fmt.Errorf("npy: unsupported Go element type %T", z)
	}
	return DType{Kind: kind, ItemSize: size, Order: hostOrder, Descr: formatDescr(kind, size)}, nil
}

// formatDescr builds a descr string in the host byte order.
func formatDescr(kind byte, size int) string {
	var prefix byte = '|' // single-byte types use '|'
	if size > 1 {
		if hostIsLittle {
			prefix = '<'
		} else {
			prefix = '>'
		}
	}
	return fmt.Sprintf("%c%c%d", prefix, kind, size)
}

// swapBytes byte-swaps the buffer in place according to the dtype, converting
// between little- and big-endian. For complex types each of the two
// floating-point components is swapped independently.
func swapBytes(buf []byte, dt DType) {
	unit := dt.ItemSize
	if dt.Kind == 'c' {
		unit = dt.ItemSize / 2 // swap real and imaginary parts separately
	}
	if unit <= 1 {
		return
	}
	for off := 0; off+unit <= len(buf); off += unit {
		for i, j := off, off+unit-1; i < j; i, j = i+1, j-1 {
			buf[i], buf[j] = buf[j], buf[i]
		}
	}
}
