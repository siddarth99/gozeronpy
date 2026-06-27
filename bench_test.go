package npy_test

import (
	"os"
	"path/filepath"
	"testing"

	npy "github.com/siddarth99/gonpz"
)

// makeBenchFile writes an N-element float64 array and returns its path.
func makeBenchFile(b *testing.B, n int) string {
	b.Helper()
	data := make([]float64, n)
	for i := range data {
		data[i] = float64(i)
	}
	path := filepath.Join(b.TempDir(), "bench.npy")
	if err := npy.Save(path, data, []int{n}); err != nil {
		b.Fatal(err)
	}
	return path
}

// 8M float64 = 64 MiB of array data.
const benchN = 8 << 20

func BenchmarkLoadInMemory(b *testing.B) {
	path := makeBenchFile(b, benchN)
	fi, _ := os.Stat(path)
	b.SetBytes(fi.Size())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nd, err := npy.Load[float64](path, npy.WithMode(npy.InMemory))
		if err != nil {
			b.Fatal(err)
		}
		sink(nd.Values)
	}
}

func BenchmarkLoadMmap(b *testing.B) {
	path := makeBenchFile(b, benchN)
	fi, _ := os.Stat(path)
	b.SetBytes(fi.Size())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nd, err := npy.Load[float64](path, npy.WithMode(npy.Mmap))
		if err != nil {
			b.Fatal(err)
		}
		// Touch every element so the comparison includes page faults.
		var s float64
		for _, v := range nd.Values {
			s += v
		}
		sinkF(s)
		nd.Close()
	}
}

func BenchmarkOpenDynamic(b *testing.B) {
	path := makeBenchFile(b, benchN)
	fi, _ := os.Stat(path)
	b.SetBytes(fi.Size())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		arr, err := npy.Open(path, npy.WithMode(npy.InMemory))
		if err != nil {
			b.Fatal(err)
		}
		v, _ := arr.Float64()
		sink(v)
	}
}

func BenchmarkSave(b *testing.B) {
	data := make([]float64, benchN)
	dir := b.TempDir()
	b.SetBytes(int64(benchN * 8))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		path := filepath.Join(dir, "out.npy")
		if err := npy.Save(path, data, []int{benchN}); err != nil {
			b.Fatal(err)
		}
	}
}

var (
	sinkSlice []float64
	sinkFloat float64
)

func sink(s []float64) { sinkSlice = s }
func sinkF(f float64)  { sinkFloat = f }
