package npy_test

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"

	npy "github.com/siddarth99/gonpz"
)

// sampleArrays is the set of named arrays used by the round-trip tests.
func sampleArrays() []npy.NamedArray {
	return []npy.NamedArray{
		{Name: "a", Data: []float64{1, 2, 3, 4, 5, 6}, Shape: []int{2, 3}},
		{Name: "b", Data: []int32{1, 2, 3, 4}, Shape: nil},
		{Name: "c", Data: []bool{true, false, false, true}, Shape: []int{2, 2}},
		{Name: "d", Data: []complex128{1 + 2i, 3 + 4i}, Shape: []int{2}},
	}
}

func equalAny(t *testing.T, got *npy.Array, want any) bool {
	t.Helper()
	switch w := want.(type) {
	case []float64:
		g, _ := got.Float64()
		return slices.Equal(g, w)
	case []float32:
		g, _ := got.Float32()
		return slices.Equal(g, w)
	case []int32:
		g, _ := got.Int32()
		return slices.Equal(g, w)
	case []int64:
		g, _ := got.Int64()
		return slices.Equal(g, w)
	case []bool:
		g, _ := got.Bool()
		return slices.Equal(g, w)
	case []complex128:
		g, _ := got.Complex128()
		return slices.Equal(g, w)
	default:
		t.Fatalf("unhandled want type %T", want)
		return false
	}
}

func TestZipRoundTrip(t *testing.T) {
	arrays := sampleArrays()
	for _, compressed := range []bool{false, true} {
		for _, mode := range []npy.AccessMode{npy.InMemory, npy.Mmap} {
			name := "stored"
			if compressed {
				name = "deflate"
			}
			t.Run(name, func(t *testing.T) {
				path := filepath.Join(t.TempDir(), "arc.npz")
				if err := npy.SaveZip(path, arrays, npy.WithCompression(compressed)); err != nil {
					t.Fatalf("SaveZip: %v", err)
				}

				arc, err := npy.OpenZip(path, npy.WithMode(mode))
				if err != nil {
					t.Fatalf("OpenZip: %v", err)
				}
				defer arc.Close()

				if got := arc.Names(); !slices.Equal(got, []string{"a", "b", "c", "d"}) {
					t.Fatalf("Names = %v", got)
				}
				for _, na := range arrays {
					if !arc.Has(na.Name) {
						t.Fatalf("Has(%q) = false", na.Name)
					}
					arr, err := arc.Array(na.Name)
					if err != nil {
						t.Fatalf("Array(%q): %v", na.Name, err)
					}
					if !equalAny(t, arr, na.Data) {
						t.Fatalf("array %q mismatch: got %v want %v", na.Name, arr.Data, na.Data)
					}
				}

				// Typed accessor on a known entry.
				nd, err := npy.ZipValues[float64](arc, "a")
				if err != nil {
					t.Fatalf("ZipValues: %v", err)
				}
				if !slices.Equal(nd.Values, []float64{1, 2, 3, 4, 5, 6}) || !slices.Equal(nd.Shape, []int{2, 3}) {
					t.Fatalf("ZipValues a = %v %v", nd.Values, nd.Shape)
				}
			})
		}
	}
}

func TestZipStreamingWriter(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stream.npz")
	zw, err := npy.CreateZip(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := npy.AddTyped(zw, "vec", []float64{1, 2, 3}, nil); err != nil {
		t.Fatal(err)
	}
	if err := zw.Add("mat", []int16{1, 2, 3, 4}, []int{2, 2}); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}

	arc, err := npy.OpenZip(path)
	if err != nil {
		t.Fatal(err)
	}
	defer arc.Close()

	all, err := arc.All()
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 2 {
		t.Fatalf("len(All) = %d, want 2", len(all))
	}
	vec, _ := all["vec"].Float64()
	if !slices.Equal(vec, []float64{1, 2, 3}) {
		t.Fatalf("vec = %v", vec)
	}
	mat, _ := all["mat"].Int16()
	if !slices.Equal(mat, []int16{1, 2, 3, 4}) || !slices.Equal(all["mat"].Shape, []int{2, 2}) {
		t.Fatalf("mat = %v %v", mat, all["mat"].Shape)
	}
}

func TestZipErrors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "e.npz")
	if err := npy.SaveZip(path, []npy.NamedArray{{Name: "x", Data: []float64{1, 2}}}); err != nil {
		t.Fatal(err)
	}
	arc, err := npy.OpenZip(path)
	if err != nil {
		t.Fatal(err)
	}
	defer arc.Close()

	if _, err := arc.Array("missing"); err == nil {
		t.Fatal("expected error for missing array")
	}
	if _, err := npy.ZipValues[int32](arc, "x"); err == nil {
		t.Fatal("expected dtype mismatch error")
	}
}

// TestReadNpzGeneratedFiles verifies the Go library reads .npz archives that
// NumPy produced (both uncompressed and compressed).
func TestReadNpzGeneratedFiles(t *testing.T) {
	py := pythonWithNumpy(t)
	dir := t.TempDir()
	if out, err := exec.Command(py, "scripts/npygen.py", "genzip", dir).CombinedOutput(); err != nil {
		t.Fatalf("genzip failed: %v\n%s", err, out)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "manifest_npz.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifests []struct {
		File       string          `json:"file"`
		Compressed bool            `json:"compressed"`
		Arrays     []manifestEntry `json:"arrays"`
	}
	if err := json.Unmarshal(raw, &manifests); err != nil {
		t.Fatal(err)
	}
	if len(manifests) == 0 {
		t.Fatal("npz manifest empty")
	}

	for _, m := range manifests {
		for _, mode := range []npy.AccessMode{npy.InMemory, npy.Mmap} {
			t.Run(m.File, func(t *testing.T) {
				arc, err := npy.OpenZip(filepath.Join(dir, m.File), npy.WithMode(mode))
				if err != nil {
					t.Fatalf("OpenZip: %v", err)
				}
				defer arc.Close()
				for _, e := range m.Arrays {
					arr, err := arc.Array(e.Name)
					if err != nil {
						t.Fatalf("Array(%q): %v", e.Name, err)
					}
					if !slices.Equal(arr.Shape, e.Shape) {
						t.Errorf("%s shape = %v, want %v", e.Name, arr.Shape, e.Shape)
					}
					if arr.Dtype.ItemSize != e.ItemSize || string(arr.Dtype.Kind) != e.Kind {
						t.Errorf("%s dtype = %c%d, want %s%d", e.Name, arr.Dtype.Kind, arr.Dtype.ItemSize, e.Kind, e.ItemSize)
					}
					compareValues(t, arr, e)
				}
			})
		}
	}
}

// TestWriteNpzRoundTripWithNumpy writes a .npz with Go and verifies NumPy reads
// the same arrays.
func TestWriteNpzRoundTripWithNumpy(t *testing.T) {
	py := pythonWithNumpy(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "go.npz")
	arrays := sampleArrays()
	if err := npy.SaveZip(path, arrays, npy.WithCompression(true)); err != nil {
		t.Fatal(err)
	}

	out, err := exec.Command(py, "scripts/npygen.py", "checkzip", path).Output()
	if err != nil {
		t.Fatalf("checkzip: %v", err)
	}
	var got struct {
		Arrays []manifestEntry `json:"arrays"`
	}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("parse: %v\n%s", err, out)
	}
	byName := map[string]manifestEntry{}
	for _, e := range got.Arrays {
		byName[e.Name] = e
	}
	for _, na := range arrays {
		e, ok := byName[na.Name]
		if !ok {
			t.Errorf("numpy did not read array %q", na.Name)
			continue
		}
		switch na.Name {
		case "a":
			if !slices.Equal(e.Shape, []int{2, 3}) || e.Kind != "f" {
				t.Errorf("a = %s %v", e.Kind, e.Shape)
			}
		case "d":
			if e.Kind != "c" || e.ItemSize != 16 {
				t.Errorf("d = %s%d", e.Kind, e.ItemSize)
			}
		}
	}
}
