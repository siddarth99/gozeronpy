//go:build !unix

package npy

// mmapSupported reports whether memory mapping is available on this platform.
// On non-Unix platforms the library transparently falls back to reading the
// file into memory.
const mmapSupported = false

// mmapHandle is a stub on platforms without memory-mapping support.
type mmapHandle struct {
	data []byte
}

func mmapOpen(path string) (*mmapHandle, error) { return nil, errMmapUnsupported }

func (m *mmapHandle) Close() error { return nil }
