//go:build !darwin && !linux

package npy

// availableRAM is unknown on platforms without a specific implementation; the
// Auto access mode then falls back to a fixed size threshold.
func availableRAM() (uint64, bool) { return 0, false }
