#!/usr/bin/env python3
"""Helpers for cross-validating the Go npy library against NumPy.

Subcommands:
  gen <dir>          Write reference .npy files plus manifest.json into <dir>.
  check <file.npy>   Print JSON describing a .npy file (used to verify files
                     written by the Go library).
  genzip <dir>       Write reference .npz archives plus manifest_npz.json.
  checkzip <f.npz>   Print JSON describing the arrays in a .npz archive.
"""
import json
import os
import sys

import numpy as np

# dtype char used by NumPy's .npy descr -> our "kind"
KIND = {"f": "f", "i": "i", "u": "u", "b": "b", "c": "c"}


def kind_of(dt: np.dtype) -> str:
    return KIND[dt.kind]


def values_of(arr: np.ndarray):
    """Flatten in memory (storage) order, matching the on-disk byte order."""
    flat = arr.ravel(order="A")
    k = arr.dtype.kind
    if k == "c":
        return [[float(c.real), float(c.imag)] for c in flat]
    if k == "b":
        return [bool(x) for x in flat]
    if k == "f":
        return [float(x) for x in flat]
    return [int(x) for x in flat]


def describe(arr: np.ndarray, file: str) -> dict:
    return {
        "file": os.path.basename(file),
        "descr": arr.dtype.str,
        "kind": kind_of(arr.dtype),
        "itemsize": arr.dtype.itemsize,
        "shape": list(arr.shape),
        "fortran": bool(np.isfortran(arr)),
        "values": values_of(arr),
    }


def describe_named(arr: np.ndarray, name: str) -> dict:
    d = describe(arr, name)
    del d["file"]
    d["name"] = name
    return d


def gen(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    entries = []

    base_dtypes = [
        "f4", "f8",
        "i1", "i2", "i4", "i8",
        "u1", "u2", "u4", "u8",
        "c8", "c16",
    ]
    shapes = [(), (5,), (2, 3), (2, 3, 4)]

    idx = 0
    for dt in base_dtypes:
        for shape in shapes:
            n = int(np.prod(shape)) if shape else 1
            seq = np.arange(1, n + 1)
            if dt.startswith("c"):
                vals = (seq + 1j * (seq + 1)).astype("<" + dt)
            else:
                vals = seq.astype("<" + dt)
            # little-endian, C order
            arr = vals.reshape(shape)
            for order, suffix in (("<", "le"), (">", "be")):
                a = arr.astype(order + dt) if arr.dtype.itemsize > 1 else arr
                fname = f"a{idx:03d}_{dt}_{suffix}.npy"
                idx += 1
                np.save(os.path.join(out_dir, fname), a)
                entries.append(describe(a, fname))
            # Fortran order (little-endian) for multi-dim shapes
            if len(shape) >= 2:
                af = np.asfortranarray(arr)
                fname = f"a{idx:03d}_{dt}_fortran.npy"
                idx += 1
                np.save(os.path.join(out_dir, fname), af)
                entries.append(describe(af, fname))

    # boolean array
    barr = np.array([True, False, True, True, False, False]).reshape(2, 3)
    np.save(os.path.join(out_dir, "bool.npy"), barr)
    entries.append(describe(barr, "bool.npy"))

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(entries, f, indent=2)
    print(f"wrote {len(entries)} files to {out_dir}")


def check(path: str) -> None:
    arr = np.load(path)
    json.dump(describe(arr, path), sys.stdout)
    sys.stdout.write("\n")


def genzip(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    arrays = {
        "a": np.arange(6, dtype="<f8").reshape(2, 3),
        "b": np.arange(4, dtype="<i4"),
        "c": np.array([[True, False], [False, True]]),
        "d": (np.arange(3) + 1j * np.arange(1, 4)).astype("<c16"),
    }

    manifest = []
    for fname, compressed in (("multi.npz", False), ("multi_compressed.npz", True)):
        path = os.path.join(out_dir, fname)
        if compressed:
            np.savez_compressed(path, **arrays)
        else:
            np.savez(path, **arrays)
        manifest.append({
            "file": fname,
            "compressed": compressed,
            "arrays": [describe_named(arrays[k], k) for k in arrays],
        })

    with open(os.path.join(out_dir, "manifest_npz.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(manifest)} archives to {out_dir}")


def checkzip(path: str) -> None:
    with np.load(path) as npz:
        out = {
            "file": os.path.basename(path),
            "arrays": [describe_named(npz[name], name) for name in npz.files],
        }
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")


def main(argv):
    if len(argv) >= 3 and argv[1] == "gen":
        gen(argv[2])
    elif len(argv) >= 3 and argv[1] == "check":
        check(argv[2])
    elif len(argv) >= 3 and argv[1] == "genzip":
        genzip(argv[2])
    elif len(argv) >= 3 and argv[1] == "checkzip":
        checkzip(argv[2])
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
