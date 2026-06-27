//go:build linux

package npy

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

// availableRAM returns the memory the kernel reports as available (MemAvailable
// from /proc/meminfo), falling back to MemTotal.
func availableRAM() (uint64, bool) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, false
	}
	defer f.Close()

	var memAvailable, memTotal uint64
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if v, ok := meminfoValue(line, "MemAvailable:"); ok {
			memAvailable = v
		} else if v, ok := meminfoValue(line, "MemTotal:"); ok {
			memTotal = v
		}
	}
	switch {
	case memAvailable > 0:
		return memAvailable, true
	case memTotal > 0:
		return memTotal, true
	default:
		return 0, false
	}
}

// meminfoValue parses a "Key:   12345 kB" line, returning bytes.
func meminfoValue(line, key string) (uint64, bool) {
	if !strings.HasPrefix(line, key) {
		return 0, false
	}
	fields := strings.Fields(line[len(key):])
	if len(fields) == 0 {
		return 0, false
	}
	kb, err := strconv.ParseUint(fields[0], 10, 64)
	if err != nil {
		return 0, false
	}
	return kb * 1024, true
}
