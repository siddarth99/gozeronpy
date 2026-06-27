//go:build darwin

package npy

import (
	"syscall"
	"unsafe"
)

// availableRAM returns the machine's physical memory in bytes. macOS does not
// expose a cheap "available" figure, so total physical memory is used as the
// budget together with Options.MaxRAMFraction.
func availableRAM() (uint64, bool) {
	// sysctl(CTL_HW, HW_MEMSIZE) -> uint64 bytes of physical memory.
	mib := [2]int32{6 /* CTL_HW */, 24 /* HW_MEMSIZE */}
	var out uint64
	n := uintptr(unsafe.Sizeof(out))
	_, _, errno := syscall.Syscall6(
		syscall.SYS___SYSCTL,
		uintptr(unsafe.Pointer(&mib[0])), 2,
		uintptr(unsafe.Pointer(&out)), uintptr(unsafe.Pointer(&n)),
		0, 0,
	)
	if errno != 0 || out == 0 {
		return 0, false
	}
	return out, true
}
