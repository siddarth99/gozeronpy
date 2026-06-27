// Command example demonstrates reading and writing .npy files.
//
//	go run ./examples
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	npy "github.com/siddarth99/gonpz"
)

func main() {
	dir, err := os.MkdirTemp("", "npyexample")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "matrix.npy")

	// Write a 2x3 float64 matrix.
	if err := npy.Save(path, []float64{1, 2, 3, 4, 5, 6}, []int{2, 3}); err != nil {
		log.Fatal(err)
	}

	// Read it back dynamically — shape and dtype are auto-detected.
	arr, err := npy.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer arr.Close()
	fmt.Printf("shape=%v dtype=%s (%s)\n", arr.Shape, arr.Dtype, arr.Dtype.GoType())

	xs, _ := arr.Float64()
	fmt.Println("data:", xs)

	// Read it back with a known type (fastest path).
	nd, err := npy.Load[float64](path)
	if err != nil {
		log.Fatal(err)
	}
	defer nd.Close()
	fmt.Printf("typed: shape=%v values=%v\n", nd.Shape, nd.Values)

	// Index helper for a row-major 2-D array.
	get := func(r, c int) float64 { return nd.Values[r*nd.Shape[1]+c] }
	fmt.Printf("element [1][2] = %v\n", get(1, 2))
}
