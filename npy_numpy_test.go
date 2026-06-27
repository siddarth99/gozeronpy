package npy_test

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"

	npy "github.com/siddarth99/gozeronpy"
)

// manifestEntry mirrors the JSON emitted by scripts/npygen.py.
type manifestEntry struct {
	File     string          `json:"file"`
	Name     string          `json:"name"`
	Descr    string          `json:"descr"`
	Kind     string          `json:"kind"`
	ItemSize int             `json:"itemsize"`
	Shape    []int           `json:"shape"`
	Fortran  bool            `json:"fortran"`
	Values   json.RawMessage `json:"values"`
}

// pythonWithNumpy returns the python3 interpreter if numpy is importable.
func pythonWithNumpy(t *testing.T) string {
	t.Helper()
	py, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not found; skipping NumPy cross-validation")
	}
	if err := exec.Command(py, "-c", "import numpy").Run(); err != nil {
		t.Skip("numpy not importable; skipping NumPy cross-validation")
	}
	return py
}

// TestReadNumpyGeneratedFiles generates reference .npy files with NumPy and
// verifies the Go library reads identical shapes, dtypes and values.
func TestReadNumpyGeneratedFiles(t *testing.T) {
	py := pythonWithNumpy(t)
	dir := t.TempDir()
	if out, err := exec.Command(py, "scripts/npygen.py", "gen", dir).CombinedOutput(); err != nil {
		t.Fatalf("npygen gen failed: %v\n%s", err, out)
	}

	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var entries []manifestEntry
	if err := json.Unmarshal(raw, &entries); err != nil {
		t.Fatal(err)
	}
	if len(entries) == 0 {
		t.Fatal("manifest is empty")
	}

	for _, e := range entries {
		t.Run(e.File, func(t *testing.T) {
			arr, err := npy.Open(filepath.Join(dir, e.File))
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			defer arr.Close()

			if !slices.Equal(arr.Shape, e.Shape) {
				t.Errorf("shape = %v, want %v", arr.Shape, e.Shape)
			}
			if arr.Fortran != e.Fortran {
				t.Errorf("fortran = %v, want %v", arr.Fortran, e.Fortran)
			}
			if string(arr.Dtype.Kind) != e.Kind {
				t.Errorf("kind = %c, want %s", arr.Dtype.Kind, e.Kind)
			}
			if arr.Dtype.ItemSize != e.ItemSize {
				t.Errorf("itemsize = %d, want %d", arr.Dtype.ItemSize, e.ItemSize)
			}
			compareValues(t, arr, e)
		})
	}
}

// TestWriteRoundTripWithNumpy writes files with Go and verifies NumPy reads
// back the same shapes, dtypes and values.
func TestWriteRoundTripWithNumpy(t *testing.T) {
	py := pythonWithNumpy(t)
	dir := t.TempDir()

	type tc struct {
		name  string
		write func(path string) error
		check manifestEntry
	}
	cases := []tc{
		{"f8", func(p string) error { return npy.Save(p, []float64{1, 2, 3, 4, 5, 6}, []int{2, 3}) },
			manifestEntry{Kind: "f", ItemSize: 8, Shape: []int{2, 3}}},
		{"f4", func(p string) error { return npy.Save(p, []float32{1, 2, 3}, []int{3}) },
			manifestEntry{Kind: "f", ItemSize: 4, Shape: []int{3}}},
		{"i4", func(p string) error { return npy.Save(p, []int32{-1, 0, 1, 2}, []int{2, 2}) },
			manifestEntry{Kind: "i", ItemSize: 4, Shape: []int{2, 2}}},
		{"u8", func(p string) error { return npy.Save(p, []uint64{1, 2, 3}, nil) },
			manifestEntry{Kind: "u", ItemSize: 8, Shape: []int{3}}},
		{"c16", func(p string) error { return npy.Save(p, []complex128{1 + 2i, 3 + 4i}, []int{2}) },
			manifestEntry{Kind: "c", ItemSize: 16, Shape: []int{2}}},
		{"bool", func(p string) error { return npy.Save(p, []bool{true, false, true, true}, []int{2, 2}) },
			manifestEntry{Kind: "b", ItemSize: 1, Shape: []int{2, 2}}},
		{"fortran", func(p string) error {
			return npy.Save(p, []float64{1, 2, 3, 4}, []int{2, 2}, npy.WithFortran(true))
		}, manifestEntry{Kind: "f", ItemSize: 8, Shape: []int{2, 2}, Fortran: true}},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			path := filepath.Join(dir, c.name+".npy")
			if err := c.write(path); err != nil {
				t.Fatalf("write: %v", err)
			}
			out, err := exec.Command(py, "scripts/npygen.py", "check", path).Output()
			if err != nil {
				t.Fatalf("npygen check: %v", err)
			}
			var got manifestEntry
			if err := json.Unmarshal(out, &got); err != nil {
				t.Fatalf("parse check output: %v\n%s", err, out)
			}
			if got.Kind != c.check.Kind || got.ItemSize != c.check.ItemSize {
				t.Errorf("dtype = %s%d, want %s%d", got.Kind, got.ItemSize, c.check.Kind, c.check.ItemSize)
			}
			if !slices.Equal(got.Shape, c.check.Shape) {
				t.Errorf("shape = %v, want %v", got.Shape, c.check.Shape)
			}
			if got.Fortran != c.check.Fortran {
				t.Errorf("fortran = %v, want %v", got.Fortran, c.check.Fortran)
			}
		})
	}
}

// compareValues checks the array's flat data against the manifest's values,
// dispatching on the dtype kind.
func compareValues(t *testing.T, arr *npy.Array, e manifestEntry) {
	t.Helper()
	switch e.Kind {
	case "b":
		var want []bool
		mustJSON(t, e.Values, &want)
		got, err := arr.Bool()
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Errorf("bool data = %v, want %v", got, want)
		}
	case "c":
		var want [][2]float64
		mustJSON(t, e.Values, &want)
		if e.ItemSize == 8 {
			got, err := arr.Complex64()
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(want) {
				t.Fatalf("len = %d, want %d", len(got), len(want))
			}
			for i, c := range got {
				if float64(real(c)) != want[i][0] || float64(imag(c)) != want[i][1] {
					t.Errorf("elem %d = %v, want %v", i, c, want[i])
				}
			}
		} else {
			got, err := arr.Complex128()
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(want) {
				t.Fatalf("len = %d, want %d", len(got), len(want))
			}
			for i, c := range got {
				if real(c) != want[i][0] || imag(c) != want[i][1] {
					t.Errorf("elem %d = %v, want %v", i, c, want[i])
				}
			}
		}
	default: // f, i, u
		var want []float64
		mustJSON(t, e.Values, &want)
		got, err := arr.AsFloat64()
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Errorf("data = %v, want %v", got, want)
		}
	}
}

func mustJSON(t *testing.T, raw json.RawMessage, v any) {
	t.Helper()
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatalf("unmarshal values: %v", err)
	}
}
