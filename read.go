package npy

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"unsafe"
)

// AccessMode controls how a file's data region is brought into the process.
type AccessMode int

const (
	// Auto picks between InMemory and Mmap based on the data size relative to
	// available system memory. This is the default.
	Auto AccessMode = iota
	// InMemory reads the whole file into a heap buffer.
	InMemory
	// Mmap memory-maps the file (falling back to InMemory where unsupported).
	Mmap
)

// autoMmapThreshold is the data size above which Auto switches to mmap when
// system memory cannot be determined.
const autoMmapThreshold = 512 << 20 // 512 MiB

var (
	errMmapEmpty       = errors.New("npy: cannot mmap an empty file")
	errMmapUnsupported = errors.New("npy: mmap not supported on this platform")
)

// config holds resolved options.
type config struct {
	mode           AccessMode
	maxRAMFraction float64
	fortran        bool // write-side: store in Fortran order
	compress       bool // write-side: deflate .npz entries
}

// Option customises read and write behaviour.
type Option func(*config)

// WithMode selects the access mode used when reading from a file path.
func WithMode(m AccessMode) Option { return func(c *config) { c.mode = m } }

// WithMaxRAMFraction sets the fraction of system memory above which Auto mode
// switches from reading into memory to memory-mapping. The default is 0.5.
func WithMaxRAMFraction(f float64) Option { return func(c *config) { c.maxRAMFraction = f } }

// WithFortran requests Fortran (column-major) ordering when writing.
func WithFortran(fortran bool) Option { return func(c *config) { c.fortran = fortran } }

// WithCompression enables DEFLATE compression of .npz archive entries (like
// numpy.savez_compressed). It has no effect on single .npy files. The default
// is no compression (like numpy.savez).
func WithCompression(compress bool) Option { return func(c *config) { c.compress = compress } }

func resolve(opts []Option) config {
	c := config{mode: Auto, maxRAMFraction: 0.5}
	for _, o := range opts {
		if o != nil {
			o(&c)
		}
	}
	if c.maxRAMFraction <= 0 {
		c.maxRAMFraction = 0.5
	}
	return c
}

// Open reads a .npy file, returning a dynamically-typed Array whose shape and
// element type are taken from the file. This is the most NumPy-like entry
// point:
//
//	arr, err := npy.Open("data.npy")
//	fmt.Println(arr.Shape)
//
// For memory-mapped arrays, call Close when done.
func Open(path string, opts ...Option) (*Array, error) {
	cfg := resolve(opts)
	useMmap, err := decideMmap(path, cfg)
	if err != nil {
		return nil, err
	}
	if useMmap {
		if arr, err := openMmapped(path); err == nil {
			return arr, nil
		} else if !errors.Is(err, errMmapEmpty) {
			return nil, err
		}
		// empty data region: fall through to the in-memory path.
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Decode(f)
}

// Decode reads a .npy stream into a dynamically-typed Array. The whole array
// is read into memory.
func Decode(r io.Reader) (*Array, error) {
	br := bufferedReader(r)
	h, err := readHeader(br)
	if err != nil {
		return nil, err
	}
	buf, err := readData(br, h)
	if err != nil {
		return nil, err
	}
	return buildArray(h, buf, true), nil
}

// Load reads a .npy file into a statically-typed NDArray[T]. T must match the
// file's element kind and size (byte order is converted automatically). This
// is the fastest entry point. For memory-mapped arrays, call Close when done.
func Load[T Element](path string, opts ...Option) (*NDArray[T], error) {
	cfg := resolve(opts)
	want, err := dtypeFor[T]()
	if err != nil {
		return nil, err
	}
	useMmap, err := decideMmap(path, cfg)
	if err != nil {
		return nil, err
	}
	if useMmap {
		if nd, err := loadMmapped[T](path, want); err == nil {
			return nd, nil
		} else if !errors.Is(err, errMmapEmpty) {
			return nil, err
		}
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read[T](f)
}

// Read reads a .npy stream into a statically-typed NDArray[T].
func Read[T Element](r io.Reader) (*NDArray[T], error) {
	want, err := dtypeFor[T]()
	if err != nil {
		return nil, err
	}
	br := bufferedReader(r)
	h, err := readHeader(br)
	if err != nil {
		return nil, err
	}
	if err := checkDtype(h.dtype, want); err != nil {
		return nil, err
	}
	buf, err := readData(br, h)
	if err != nil {
		return nil, err
	}
	return buildTyped[T](h, buf, true), nil
}

// readData reads the array's data region into a freshly-allocated, 8-byte
// aligned buffer.
func readData(r io.Reader, h header) ([]byte, error) {
	need := numElements(h.shape) * h.dtype.ItemSize
	buf := alignedBuffer(need)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, fmt.Errorf("npy: reading data: %w", err)
	}
	return buf, nil
}

// openMmapped maps the file and builds a dynamically-typed Array. The Array's
// Data may alias the mapping (zero-copy); Close unmaps it.
func openMmapped(path string) (*Array, error) {
	m, h, region, err := mapAndLocate(path)
	if err != nil {
		return nil, err
	}
	buf, copied := normalizeData(region, h.dtype, false)
	arr := &Array{Shape: h.shape, Fortran: h.fortran, Dtype: h.dtype, Data: sliceFromBytes(buf, h.dtype)}
	if copied {
		m.Close() // data no longer aliases the mapping
	} else {
		arr.closer = m.Close
	}
	return arr, nil
}

// loadMmapped maps the file and builds a statically-typed NDArray[T].
func loadMmapped[T Element](path string, want DType) (*NDArray[T], error) {
	m, h, region, err := mapAndLocate(path)
	if err != nil {
		return nil, err
	}
	if err := checkDtype(h.dtype, want); err != nil {
		m.Close()
		return nil, err
	}
	buf, copied := normalizeData(region, h.dtype, false)
	nd := &NDArray[T]{Values: bytesAsSlice[T](buf), Shape: h.shape, Fortran: h.fortran}
	if copied {
		m.Close()
	} else {
		nd.closer = m.Close
	}
	return nd, nil
}

// normalizeData returns the array bytes in host byte order and suitably
// aligned for reinterpretation. owned indicates the buffer may be mutated in
// place (e.g. a freshly read buffer); when owned is false (a view into an
// mmap) and a transformation is required, a new aligned buffer is allocated
// and copied is reported true.
func normalizeData(data []byte, dt DType, owned bool) (buf []byte, copied bool) {
	if dt.needSwap() {
		if owned {
			swapBytes(data, dt)
			return data, false
		}
		b := alignedBuffer(len(data))
		copy(b, data)
		swapBytes(b, dt)
		return b, true
	}
	if !owned && !aligned(data, requiredAlign(dt)) {
		b := alignedBuffer(len(data))
		copy(b, data)
		return b, true
	}
	return data, false
}

// buildArray builds a dynamically-typed Array from the data region.
func buildArray(h header, data []byte, owned bool) *Array {
	buf, _ := normalizeData(data, h.dtype, owned)
	return &Array{Shape: h.shape, Fortran: h.fortran, Dtype: h.dtype, Data: sliceFromBytes(buf, h.dtype)}
}

// buildTyped builds a statically-typed NDArray[T] from the data region.
func buildTyped[T Element](h header, data []byte, owned bool) *NDArray[T] {
	buf, _ := normalizeData(data, h.dtype, owned)
	return &NDArray[T]{Values: bytesAsSlice[T](buf), Shape: h.shape, Fortran: h.fortran}
}

// mapAndLocate maps the file, parses its header, and returns the data region
// slice (a view into the mapping).
func mapAndLocate(path string) (*mmapHandle, header, []byte, error) {
	m, err := mmapOpen(path)
	if err != nil {
		return nil, header{}, nil, err
	}
	h, err := readHeader(bytes.NewReader(m.data))
	if err != nil {
		m.Close()
		return nil, header{}, nil, err
	}
	need := numElements(h.shape) * h.dtype.ItemSize
	if h.dataOffset+need > len(m.data) {
		m.Close()
		return nil, header{}, nil, fmt.Errorf("npy: data truncated: file has %d bytes, need %d", len(m.data), h.dataOffset+need)
	}
	if need == 0 {
		m.Close()
		return nil, header{}, nil, errMmapEmpty
	}
	return m, h, m.data[h.dataOffset : h.dataOffset+need], nil
}

// decideMmap returns whether to memory-map the file at path under cfg.
func decideMmap(path string, cfg config) (bool, error) {
	switch cfg.mode {
	case InMemory:
		return false, nil
	case Mmap:
		return mmapSupported, nil
	default: // Auto
		if !mmapSupported {
			return false, nil
		}
		fi, err := os.Stat(path)
		if err != nil {
			return false, err
		}
		size := fi.Size()
		ram, ok := availableRAM()
		if !ok {
			return size > autoMmapThreshold, nil
		}
		return float64(size) > float64(ram)*cfg.maxRAMFraction, nil
	}
}

// checkDtype verifies that the file's element kind and size match the
// requested Go type. Byte order may differ and is converted on read.
func checkDtype(have, want DType) error {
	if have.Kind != want.Kind || have.ItemSize != want.ItemSize {
		return fmt.Errorf("npy: file dtype is %s (%s) but %s was requested", have, have.GoType(), want.GoType())
	}
	return nil
}

// sliceFromBytes reinterprets buf as the Go slice type matching dt. The bytes
// must already be in host order and suitably aligned.
func sliceFromBytes(buf []byte, dt DType) any {
	switch dt.key() {
	case key('f', 4):
		return bytesAsSlice[float32](buf)
	case key('f', 8):
		return bytesAsSlice[float64](buf)
	case key('i', 1):
		return bytesAsSlice[int8](buf)
	case key('i', 2):
		return bytesAsSlice[int16](buf)
	case key('i', 4):
		return bytesAsSlice[int32](buf)
	case key('i', 8):
		return bytesAsSlice[int64](buf)
	case key('u', 1):
		return bytesAsSlice[uint8](buf)
	case key('u', 2):
		return bytesAsSlice[uint16](buf)
	case key('u', 4):
		return bytesAsSlice[uint32](buf)
	case key('u', 8):
		return bytesAsSlice[uint64](buf)
	case key('b', 1):
		return bytesAsSlice[bool](buf)
	case key('c', 8):
		return bytesAsSlice[complex64](buf)
	case key('c', 16):
		return bytesAsSlice[complex128](buf)
	}
	return nil
}

// alignedBuffer allocates a byte buffer of length n whose first byte is
// 8-byte aligned, so it can safely be reinterpreted as any supported element
// type.
func alignedBuffer(n int) []byte {
	if n == 0 {
		return nil
	}
	backing := make([]uint64, (n+7)/8)
	return sliceAsBytes(backing)[:n]
}

// aligned reports whether the first byte of b meets the given alignment.
func aligned(b []byte, align int) bool {
	if len(b) == 0 || align <= 1 {
		return true
	}
	return uintptr(unsafe.Pointer(&b[0]))%uintptr(align) == 0
}

// requiredAlign returns the alignment a buffer must satisfy to be cast to the
// Go type for dt.
func requiredAlign(dt DType) int {
	a := dt.ItemSize
	if dt.Kind == 'c' {
		a = dt.ItemSize / 2 // complex aligns to its component
	}
	if a > 8 {
		a = 8
	}
	return a
}
