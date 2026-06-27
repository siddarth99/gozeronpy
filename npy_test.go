package npy_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	npy "github.com/siddarth99/gonpz"
)

// roundTrip writes data with the given shape, reads it back through every
// public reader, and checks the values and shape survive.
func roundTrip[T npy.Element](t *testing.T, data []T, shape []int) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "x.npy")
	if err := npy.Save(path, data, shape, npy.WithMode(npy.InMemory)); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Dynamic file read.
	arr, err := npy.Open(path, npy.WithMode(npy.InMemory))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if !slices.Equal(arr.Shape, shape) {
		t.Fatalf("Open shape = %v, want %v", arr.Shape, shape)
	}
	got, err := npy.Values[T](arr)
	if err != nil {
		t.Fatalf("Values: %v", err)
	}
	if !slices.Equal(got, data) {
		t.Fatalf("Open data = %v, want %v", got, data)
	}
	arr.Close()

	// Typed file read.
	nd, err := npy.Load[T](path, npy.WithMode(npy.InMemory))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !slices.Equal(nd.Values, data) {
		t.Fatalf("Load data = %v, want %v", nd.Values, data)
	}
	if !slices.Equal(nd.Shape, shape) {
		t.Fatalf("Load shape = %v, want %v", nd.Shape, shape)
	}
	nd.Close()

	// Memory-mapped typed read.
	ndm, err := npy.Load[T](path, npy.WithMode(npy.Mmap))
	if err != nil {
		t.Fatalf("Load(mmap): %v", err)
	}
	if !slices.Equal(ndm.Values, data) {
		t.Fatalf("Load(mmap) data = %v, want %v", ndm.Values, data)
	}
	ndm.Close()

	// Stream read.
	var buf bytes.Buffer
	if err := npy.Write(&buf, data, shape); err != nil {
		t.Fatalf("Write: %v", err)
	}
	nds, err := npy.Read[T](&buf)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if !slices.Equal(nds.Values, data) {
		t.Fatalf("Read data = %v, want %v", nds.Values, data)
	}
}

func TestRoundTripAllTypes(t *testing.T) {
	roundTrip(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	roundTrip(t, []float64{1.5, -2.5, 3.25, 4}, []int{4})
	roundTrip(t, []int8{-1, 2, -3, 4}, []int{2, 2})
	roundTrip(t, []int16{-100, 200, -300}, []int{3})
	roundTrip(t, []int32{1, 2, 3, 4, 5, 6}, []int{1, 6})
	roundTrip(t, []int64{1 << 40, -(1 << 40)}, []int{2})
	roundTrip(t, []uint8{0, 255, 128}, []int{3})
	roundTrip(t, []uint16{0, 65535}, []int{2})
	roundTrip(t, []uint32{0, 4294967295}, []int{2})
	roundTrip(t, []uint64{0, 1 << 63}, []int{2})
	roundTrip(t, []bool{true, false, true, true}, []int{2, 2})
	roundTrip(t, []complex64{1 + 2i, 3 + 4i}, []int{2})
	roundTrip(t, []complex128{1 + 2i, -3 - 4i, 5 + 6i}, []int{3})
}

func TestNilShapeIs1D(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.npy")
	if err := npy.Save(path, []float64{1, 2, 3}, nil); err != nil {
		t.Fatal(err)
	}
	arr, err := npy.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer arr.Close()
	if !slices.Equal(arr.Shape, []int{3}) {
		t.Fatalf("shape = %v, want [3]", arr.Shape)
	}
}

func TestScalarZeroDim(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.npy")
	if err := npy.Save(path, []float64{42}, []int{}); err != nil {
		t.Fatal(err)
	}
	arr, err := npy.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer arr.Close()
	if len(arr.Shape) != 0 {
		t.Fatalf("shape = %v, want []", arr.Shape)
	}
	if arr.Len() != 1 {
		t.Fatalf("Len = %d, want 1", arr.Len())
	}
	v, _ := arr.Float64()
	if len(v) != 1 || v[0] != 42 {
		t.Fatalf("data = %v, want [42]", v)
	}
}

func TestHeaderAlignment(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.npy")
	if err := npy.Save(path, []float64{1, 2, 3, 4}, []int{2, 2}); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// magic(6)+version(2)+lenfield(2)+headerlen must be a multiple of 64.
	headerLen := int(binary.LittleEndian.Uint16(raw[8:10]))
	total := 10 + headerLen
	if total%64 != 0 {
		t.Fatalf("header end offset %d is not 64-aligned", total)
	}
	if raw[total-1] != '\n' {
		t.Fatalf("header does not end with newline")
	}
}

func TestDtypeMismatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.npy")
	if err := npy.Save(path, []float64{1, 2, 3}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := npy.Load[int32](path); err == nil {
		t.Fatal("expected dtype mismatch error, got nil")
	}
}

func TestReadBigEndian(t *testing.T) {
	// Build a >f8 (big-endian float64) file by hand and read it.
	path := filepath.Join(t.TempDir(), "be.npy")
	want := []float64{1.5, 2.5, 3.5}
	dataBE := make([]byte, 8*len(want))
	for i, v := range want {
		binary.BigEndian.PutUint64(dataBE[i*8:], math.Float64bits(v))
	}
	writeRawNpy(t, path, ">f8", []int{3}, false, dataBE)

	nd, err := npy.Load[float64](path, npy.WithMode(npy.InMemory))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(nd.Values, want) {
		t.Fatalf("got %v, want %v", nd.Values, want)
	}

	// Also exercise the mmap path, which must copy+swap rather than alias.
	ndm, err := npy.Load[float64](path, npy.WithMode(npy.Mmap))
	if err != nil {
		t.Fatal(err)
	}
	defer ndm.Close()
	if !slices.Equal(ndm.Values, want) {
		t.Fatalf("mmap got %v, want %v", ndm.Values, want)
	}
}

func TestFortranFlagPreserved(t *testing.T) {
	path := filepath.Join(t.TempDir(), "f.npy")
	if err := npy.Save(path, []float64{1, 2, 3, 4}, []int{2, 2}, npy.WithFortran(true)); err != nil {
		t.Fatal(err)
	}
	arr, err := npy.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer arr.Close()
	if !arr.Fortran {
		t.Fatal("expected Fortran flag set")
	}
}

func TestAsFloat64(t *testing.T) {
	path := filepath.Join(t.TempDir(), "i.npy")
	if err := npy.Save(path, []int32{1, 2, 3, 4}, []int{4}); err != nil {
		t.Fatal(err)
	}
	arr, err := npy.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer arr.Close()
	f, err := arr.AsFloat64()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(f, []float64{1, 2, 3, 4}) {
		t.Fatalf("AsFloat64 = %v", f)
	}
}

func TestEmptyDimension(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.npy")
	if err := npy.Save(path, []float64{}, []int{0, 3}); err != nil {
		t.Fatal(err)
	}
	arr, err := npy.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer arr.Close()
	if !slices.Equal(arr.Shape, []int{0, 3}) {
		t.Fatalf("shape = %v, want [0 3]", arr.Shape)
	}
	if arr.Len() != 0 {
		t.Fatalf("Len = %d, want 0", arr.Len())
	}
	v, err := arr.Float64()
	if err != nil || len(v) != 0 {
		t.Fatalf("data = %v err = %v, want empty", v, err)
	}
}

// TestLargeHeaderVersion2 forces a header bigger than 64 KiB, which the writer
// must encode using .npy format version 2.0 (4-byte length field).
func TestLargeHeaderVersion2(t *testing.T) {
	shape := make([]int, 30000) // ~90 KiB of "1, " in the header
	for i := range shape {
		shape[i] = 1
	}
	path := filepath.Join(t.TempDir(), "v2.npy")
	if err := npy.Save(path, []float64{42}, shape); err != nil {
		t.Fatal(err)
	}
	// Byte 6 of the file is the major version; it must be 2.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if raw[6] != 2 {
		t.Fatalf("major version = %d, want 2", raw[6])
	}
	nd, err := npy.Load[float64](path)
	if err != nil {
		t.Fatal(err)
	}
	defer nd.Close()
	if len(nd.Shape) != 30000 || nd.Values[0] != 42 {
		t.Fatalf("shape len = %d, value = %v", len(nd.Shape), nd.Values[0])
	}
}

func TestNotNpyFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.npy")
	if err := os.WriteFile(path, []byte("not a numpy file at all"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := npy.Open(path); err == nil {
		t.Fatal("expected error for non-npy file")
	}
}

// --- helpers ---

// writeRawNpy writes a .npy file with an explicit descr and raw (already
// byte-ordered) data, to construct files the Go writer would not produce
// (e.g. big-endian).
func writeRawNpy(t *testing.T, path, descr string, shape []int, fortran bool, data []byte) {
	t.Helper()
	fo := "False"
	if fortran {
		fo = "True"
	}
	shapeStr := "("
	for i, d := range shape {
		if i > 0 {
			shapeStr += ", "
		}
		shapeStr += itoa(d)
		if len(shape) == 1 {
			shapeStr += ","
		}
	}
	shapeStr += ")"
	dict := "{'descr': '" + descr + "', 'fortran_order': " + fo + ", 'shape': " + shapeStr + ", }"
	prefix := 10
	total := prefix + len(dict) + 1
	pad := (64 - total%64) % 64
	headerLen := len(dict) + pad + 1

	var buf bytes.Buffer
	buf.Write([]byte{0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0})
	var lf [2]byte
	binary.LittleEndian.PutUint16(lf[:], uint16(headerLen))
	buf.Write(lf[:])
	buf.WriteString(dict)
	for i := 0; i < pad; i++ {
		buf.WriteByte(' ')
	}
	buf.WriteByte('\n')
	buf.Write(data)
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
