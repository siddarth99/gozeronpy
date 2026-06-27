//go:build unix

package npy

import (
	"os"
	"runtime"
	"syscall"
)

// mmapSupported reports whether memory mapping is available on this platform.
const mmapSupported = true

// mmapHandle owns a memory-mapped file region.
type mmapHandle struct {
	data []byte // the entire mapped file
	file *os.File
}

// mmapOpen maps the whole file read-only into memory.
func mmapOpen(path string) (*mmapHandle, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	size := fi.Size()
	if size <= 0 {
		f.Close()
		return nil, errMmapEmpty
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, err
	}
	m := &mmapHandle{data: data, file: f}
	// Safety net: unmap if the caller forgets to Close and the handle becomes
	// unreachable. Any slice aliasing the mapping keeps the handle reachable.
	runtime.SetFinalizer(m, (*mmapHandle).Close)
	return m, nil
}

// Close unmaps the region and closes the underlying file. It is safe to call
// more than once.
func (m *mmapHandle) Close() error {
	if m.data == nil {
		return nil
	}
	runtime.SetFinalizer(m, nil)
	err := syscall.Munmap(m.data)
	if cerr := m.file.Close(); err == nil {
		err = cerr
	}
	m.data = nil
	return err
}
