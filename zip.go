package npy

import (
	"archive/zip"
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

// Archive is a read handle for a NumPy .npz file (a ZIP archive of .npy
// entries). Array names are the entry names with the ".npy" suffix removed,
// matching numpy.load(...).files.
//
//	arc, _ := npy.OpenZip("data.npz")
//	defer arc.Close()
//	for _, name := range arc.Names() {
//	    arr, _ := arc.Array(name)
//	    fmt.Println(name, arr.Shape)
//	}
type Archive struct {
	zr    *zip.Reader
	rc    *zip.ReadCloser // non-nil when opened from a file path
	mm    *mmapHandle     // non-nil when the archive is memory-mapped
	files map[string]*zip.File
	names []string
}

// OpenZip opens a .npz archive from a file path. Like Open, it may memory-map
// the file (honouring WithMode / WithMaxRAMFraction) so that uncompressed
// entries can be read zero-copy. Call Close when done.
func OpenZip(path string, opts ...Option) (*Archive, error) {
	cfg := resolve(opts)
	useMmap, err := decideMmap(path, cfg)
	if err != nil {
		return nil, err
	}
	if useMmap {
		if m, err := mmapOpen(path); err == nil {
			zr, err := zip.NewReader(bytes.NewReader(m.data), int64(len(m.data)))
			if err != nil {
				m.Close()
				return nil, err
			}
			return newArchive(zr, nil, m), nil
		} else if !errors.Is(err, errMmapEmpty) && !errors.Is(err, errMmapUnsupported) {
			return nil, err
		}
		// fall through to the file-based reader
	}
	rc, err := zip.OpenReader(path)
	if err != nil {
		return nil, err
	}
	return newArchive(&rc.Reader, rc, nil), nil
}

// ReadZip reads a .npz archive from r. size is the total length of the archive.
// Entries are read into memory (no mmap).
func ReadZip(r io.ReaderAt, size int64) (*Archive, error) {
	zr, err := zip.NewReader(r, size)
	if err != nil {
		return nil, err
	}
	return newArchive(zr, nil, nil), nil
}

func newArchive(zr *zip.Reader, rc *zip.ReadCloser, mm *mmapHandle) *Archive {
	a := &Archive{zr: zr, rc: rc, mm: mm, files: make(map[string]*zip.File, len(zr.File))}
	for _, f := range zr.File {
		name := strings.TrimSuffix(f.Name, ".npy")
		if _, exists := a.files[name]; !exists {
			a.names = append(a.names, name)
		}
		a.files[name] = f
	}
	return a
}

// Names returns the array names in the archive, in storage order.
func (a *Archive) Names() []string { return a.names }

// Has reports whether the archive contains an array with the given name.
func (a *Archive) Has(name string) bool { _, ok := a.files[name]; return ok }

// Array reads the named array into a dynamically-typed Array. For a
// memory-mapped archive the data may alias the mapping, so the returned Array
// must not be used after the Archive is closed.
func (a *Archive) Array(name string) (*Array, error) {
	f, ok := a.files[name]
	if !ok {
		return nil, fmt.Errorf("npy: array %q not found in archive", name)
	}
	h, data, owned, err := a.entryData(f)
	if err != nil {
		return nil, err
	}
	return buildArray(h, data, owned), nil
}

// ZipValues reads the named array from the archive into a statically-typed
// NDArray[T]. T must match the entry's element kind and size.
func ZipValues[T Element](a *Archive, name string) (*NDArray[T], error) {
	f, ok := a.files[name]
	if !ok {
		return nil, fmt.Errorf("npy: array %q not found in archive", name)
	}
	want, err := dtypeFor[T]()
	if err != nil {
		return nil, err
	}
	h, data, owned, err := a.entryData(f)
	if err != nil {
		return nil, err
	}
	if err := checkDtype(h.dtype, want); err != nil {
		return nil, err
	}
	return buildTyped[T](h, data, owned), nil
}

// All reads every array in the archive into a name-keyed map.
func (a *Archive) All() (map[string]*Array, error) {
	out := make(map[string]*Array, len(a.names))
	for _, name := range a.names {
		arr, err := a.Array(name)
		if err != nil {
			return nil, err
		}
		out[name] = arr
	}
	return out, nil
}

// Close releases the archive's resources (the file and/or mapping).
func (a *Archive) Close() error {
	var err error
	if a.rc != nil {
		err = a.rc.Close()
	}
	if a.mm != nil {
		if e := a.mm.Close(); err == nil {
			err = e
		}
	}
	return err
}

// entryData returns the header and the array's data-region bytes for a zip
// entry. owned reports whether the returned bytes are a private buffer (true)
// or a view into the archive mapping (false, zero-copy).
func (a *Archive) entryData(f *zip.File) (header, []byte, bool, error) {
	// Zero-copy fast path: an uncompressed entry inside a memory-mapped
	// archive can be read directly from the mapping.
	if a.mm != nil && f.Method == zip.Store {
		if off, err := f.DataOffset(); err == nil {
			end := off + int64(f.UncompressedSize64)
			if off >= 0 && end <= int64(len(a.mm.data)) {
				region := a.mm.data[off:end] // the full .npy bytes
				h, err := readHeader(bytes.NewReader(region))
				if err != nil {
					return header{}, nil, false, err
				}
				need := numElements(h.shape) * h.dtype.ItemSize
				if h.dataOffset+need > len(region) {
					return header{}, nil, false, fmt.Errorf("npy: entry %q truncated", f.Name)
				}
				return h, region[h.dataOffset : h.dataOffset+need], false, nil
			}
		}
	}

	// General path: let archive/zip read (and, if needed, inflate) the entry.
	rc, err := f.Open()
	if err != nil {
		return header{}, nil, false, err
	}
	defer rc.Close()
	br := bufio.NewReader(rc)
	h, err := readHeader(br)
	if err != nil {
		return header{}, nil, false, err
	}
	buf, err := readData(br, h)
	if err != nil {
		return header{}, nil, false, err
	}
	return h, buf, true, nil
}

// NamedArray pairs an array name with its data and shape for SaveZip.
type NamedArray struct {
	Name  string
	Data  any   // a supported element slice, e.g. []float64
	Shape []int // nil means a 1-D array of length len(Data)
}

// SaveZip writes a .npz archive containing the given named arrays. By default
// entries are stored uncompressed (like numpy.savez); pass
// WithCompression(true) for DEFLATE (like numpy.savez_compressed).
//
//	npy.SaveZip("out.npz", []npy.NamedArray{
//	    {Name: "x", Data: []float64{1, 2, 3, 4}, Shape: []int{2, 2}},
//	    {Name: "y", Data: []int32{5, 6, 7}},
//	})
func SaveZip(path string, arrays []NamedArray, opts ...Option) error {
	zw, err := CreateZip(path, opts...)
	if err != nil {
		return err
	}
	for _, na := range arrays {
		if err := zw.Add(na.Name, na.Data, na.Shape); err != nil {
			zw.abort()
			return err
		}
	}
	return zw.Close()
}

// ZipWriter incrementally writes arrays to a .npz archive. This is useful when
// arrays are produced one at a time and need not all be held in memory.
//
//	zw, _ := npy.CreateZip("out.npz")
//	npy.AddTyped(zw, "a", []float64{1, 2, 3}, nil)
//	zw.Add("b", []int32{4, 5, 6}, []int{3})
//	zw.Close()
type ZipWriter struct {
	f       *os.File
	zw      *zip.Writer
	method  uint16
	fortran bool
}

// CreateZip creates a .npz archive at path for incremental writing.
func CreateZip(path string, opts ...Option) (*ZipWriter, error) {
	cfg := resolve(opts)
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	method := uint16(zip.Store)
	if cfg.compress {
		method = zip.Deflate
	}
	return &ZipWriter{f: f, zw: zip.NewWriter(f), method: method, fortran: cfg.fortran}, nil
}

// Add writes a dynamically-typed array to the archive under name.
func (zw *ZipWriter) Add(name string, data any, shape []int) error {
	w, err := zw.entry(name)
	if err != nil {
		return err
	}
	return encode(w, data, shape, config{fortran: zw.fortran})
}

// AddTyped writes a statically-typed slice to the archive under name.
func AddTyped[T Element](zw *ZipWriter, name string, data []T, shape []int) error {
	w, err := zw.entry(name)
	if err != nil {
		return err
	}
	return Write(w, data, shape, WithFortran(zw.fortran))
}

func (zw *ZipWriter) entry(name string) (io.Writer, error) {
	if !strings.HasSuffix(name, ".npy") {
		name += ".npy"
	}
	return zw.zw.CreateHeader(&zip.FileHeader{Name: name, Method: zw.method})
}

// Close finalises the archive and closes the underlying file.
func (zw *ZipWriter) Close() error {
	err := zw.zw.Close()
	if cerr := zw.f.Close(); err == nil {
		err = cerr
	}
	return err
}

// abort tears down the writer after an error, leaving a best-effort cleanup.
func (zw *ZipWriter) abort() {
	zw.zw.Close()
	zw.f.Close()
}
