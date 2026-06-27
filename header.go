package npy

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// magic is the 6-byte prefix that begins every .npy file.
var magic = []byte{0x93, 'N', 'U', 'M', 'P', 'Y'}

// alignment is the byte boundary the header is padded to so that the array
// data that follows is suitably aligned.
const alignment = 64

// header holds the decoded contents of a .npy header.
type header struct {
	dtype      DType
	fortran    bool
	shape      []int
	dataOffset int // byte offset within the file where array data starts
}

// readHeader reads and decodes the .npy header from r.
func readHeader(r io.Reader) (header, error) {
	var h header

	buf := make([]byte, len(magic)+2)
	if _, err := io.ReadFull(r, buf); err != nil {
		return h, fmt.Errorf("npy: reading magic: %w", err)
	}
	if string(buf[:len(magic)]) != string(magic) {
		return h, fmt.Errorf("npy: not a .npy file (bad magic)")
	}
	major := buf[len(magic)]
	// minor version (buf[len(magic)+1]) does not affect parsing here.

	lenFieldSize := 2
	if major >= 2 {
		lenFieldSize = 4
	}
	lf := make([]byte, lenFieldSize)
	if _, err := io.ReadFull(r, lf); err != nil {
		return h, fmt.Errorf("npy: reading header length: %w", err)
	}
	var headerLen int
	if lenFieldSize == 2 {
		headerLen = int(binary.LittleEndian.Uint16(lf))
	} else {
		headerLen = int(binary.LittleEndian.Uint32(lf))
	}

	dict := make([]byte, headerLen)
	if _, err := io.ReadFull(r, dict); err != nil {
		return h, fmt.Errorf("npy: reading header: %w", err)
	}

	dt, fortran, shape, err := parseHeaderDict(string(dict))
	if err != nil {
		return h, err
	}
	h.dtype = dt
	h.fortran = fortran
	h.shape = shape
	h.dataOffset = len(magic) + 2 + lenFieldSize + headerLen
	return h, nil
}

// parseHeaderDict parses the Python-literal dict found in a .npy header.
func parseHeaderDict(s string) (DType, bool, []int, error) {
	descr, err := extractDescr(s)
	if err != nil {
		return DType{}, false, nil, err
	}
	dt, err := parseDescr(descr)
	if err != nil {
		return DType{}, false, nil, err
	}
	fortran, err := extractBool(s, "fortran_order")
	if err != nil {
		return DType{}, false, nil, err
	}
	shape, err := extractShape(s)
	if err != nil {
		return DType{}, false, nil, err
	}
	return dt, fortran, shape, nil
}

// extractDescr returns the quoted descr value from the header dict.
func extractDescr(s string) (string, error) {
	i := strings.Index(s, "'descr'")
	if i < 0 {
		return "", fmt.Errorf("npy: header missing 'descr'")
	}
	j := strings.IndexByte(s[i:], ':')
	if j < 0 {
		return "", fmt.Errorf("npy: malformed header near 'descr'")
	}
	rest := strings.TrimLeft(s[i+j+1:], " ")
	if len(rest) == 0 {
		return "", fmt.Errorf("npy: malformed 'descr' value")
	}
	if rest[0] == '[' || rest[0] == '(' {
		return "", fmt.Errorf("npy: structured/record dtypes are not supported")
	}
	if rest[0] != '\'' && rest[0] != '"' {
		return "", fmt.Errorf("npy: malformed 'descr' value")
	}
	quote := rest[0]
	end := strings.IndexByte(rest[1:], quote)
	if end < 0 {
		return "", fmt.Errorf("npy: unterminated 'descr' value")
	}
	return rest[1 : 1+end], nil
}

// extractBool returns the True/False value associated with key.
func extractBool(s, k string) (bool, error) {
	key := "'" + k + "'"
	i := strings.Index(s, key)
	if i < 0 {
		return false, fmt.Errorf("npy: header missing %q", k)
	}
	rest := s[i+len(key):]
	if j := strings.IndexByte(rest, ':'); j >= 0 {
		rest = strings.TrimLeft(rest[j+1:], " ")
	}
	switch {
	case strings.HasPrefix(rest, "True"):
		return true, nil
	case strings.HasPrefix(rest, "False"):
		return false, nil
	default:
		return false, fmt.Errorf("npy: malformed %q value", k)
	}
}

// extractShape parses the shape tuple from the header dict.
func extractShape(s string) ([]int, error) {
	i := strings.Index(s, "'shape'")
	if i < 0 {
		return nil, fmt.Errorf("npy: header missing 'shape'")
	}
	open := strings.IndexByte(s[i:], '(')
	if open < 0 {
		return nil, fmt.Errorf("npy: malformed 'shape' value")
	}
	open += i
	close := strings.IndexByte(s[open:], ')')
	if close < 0 {
		return nil, fmt.Errorf("npy: unterminated 'shape' value")
	}
	inner := s[open+1 : open+close]
	var shape []int
	for _, part := range strings.Split(inner, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		d, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("npy: malformed shape dimension %q", part)
		}
		if d < 0 {
			return nil, fmt.Errorf("npy: negative shape dimension %d", d)
		}
		shape = append(shape, d)
	}
	return shape, nil
}

// writeHeader writes the magic, version and padded header for the given
// dtype, ordering and shape.
func writeHeader(w io.Writer, dt DType, fortran bool, shape []int) error {
	dict := buildHeaderDict(dt, fortran, shape)

	// Try version 1.0 (2-byte length field) first; fall back to 2.0 if the
	// header is too large to express in 16 bits.
	major, minor := byte(1), byte(0)
	lenFieldSize := 2
	headerLen := paddedHeaderLen(len(dict), lenFieldSize)
	if headerLen > 0xFFFF {
		major = 2
		lenFieldSize = 4
		headerLen = paddedHeaderLen(len(dict), lenFieldSize)
	}

	out := make([]byte, 0, len(magic)+2+lenFieldSize+headerLen)
	out = append(out, magic...)
	out = append(out, major, minor)
	if lenFieldSize == 2 {
		out = binary.LittleEndian.AppendUint16(out, uint16(headerLen))
	} else {
		out = binary.LittleEndian.AppendUint32(out, uint32(headerLen))
	}
	out = append(out, dict...)
	// Pad with spaces and terminate with a newline so the data region begins
	// on an `alignment`-byte boundary.
	for len(out)-(len(magic)+2+lenFieldSize) < headerLen-1 {
		out = append(out, ' ')
	}
	out = append(out, '\n')

	_, err := w.Write(out)
	return err
}

// paddedHeaderLen returns the total header length (dict + padding + newline)
// such that magic + version + length-field + header is a multiple of
// alignment.
func paddedHeaderLen(dictLen, lenFieldSize int) int {
	prefix := len(magic) + 2 + lenFieldSize
	total := prefix + dictLen + 1 // +1 for the trailing newline
	pad := (alignment - total%alignment) % alignment
	return dictLen + pad + 1
}

// buildHeaderDict formats the Python-literal dict (without padding/newline).
func buildHeaderDict(dt DType, fortran bool, shape []int) string {
	fo := "False"
	if fortran {
		fo = "True"
	}
	return fmt.Sprintf("{'descr': '%s', 'fortran_order': %s, 'shape': %s, }", dt.Descr, fo, formatShape(shape))
}

// formatShape renders a shape as a Python tuple literal.
func formatShape(shape []int) string {
	switch len(shape) {
	case 0:
		return "()"
	case 1:
		return "(" + strconv.Itoa(shape[0]) + ",)"
	default:
		var b strings.Builder
		b.WriteByte('(')
		for i, d := range shape {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(strconv.Itoa(d))
		}
		b.WriteByte(')')
		return b.String()
	}
}

// bufferedReader wraps r in a bufio.Reader unless it already is one.
func bufferedReader(r io.Reader) *bufio.Reader {
	if br, ok := r.(*bufio.Reader); ok {
		return br
	}
	return bufio.NewReader(r)
}
