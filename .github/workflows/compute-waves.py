#!/usr/bin/env python3
"""Compute dependency-ordered build "waves" for a set of changed subports.

Given a list of subports changed in a PR, this restricts the MacPorts
dependency graph to *only* those subports (edges to unchanged ports are
ignored, because unchanged deps are installed as prebuilt binaries from the
archive server) and topologically sorts the result into waves.

A "wave" is a set of subports with no remaining dependency on a later wave, so
every subport in wave N depends only on subports built in waves < N. Each
downstream wave restores the archives produced by all earlier waves so a
changed dependent is built and tested against the PR version of its changed
dependencies rather than the published master binary.

Two practical adjustments for GitHub Actions:

* Waves are split into *chunks* of at most --chunk-size subports; one CI job
  builds one chunk sequentially. This amortizes the ~5 min MacPorts bootstrap
  across several small ports instead of paying it per port.

* GitHub Actions cannot create a dynamic number of chained jobs, so waves
  beyond --max-waves collapse into the last wave (preserving topological
  order). This stays correct because mpbb install-dependencies falls back to
  building a changed dependency from source when its archive is not available;
  collapsing only costs redundant work, never correctness.

Usage:
    compute-waves.py [--chunk-size N] [--max-waves N] SUBPORT [SUBPORT ...]

Reads dependencies via `port info`. Emits the wave list as JSON on stdout:
    [[{"label": "w0c0", "ports": "llvm-22"}],
     [{"label": "w1c0", "ports": "clang-22 mlir-22"}], ...]
and when run inside GitHub Actions (GITHUB_OUTPUT set) writes the same JSON as
`waves` plus `num_waves`.
"""

import argparse
import json
import os
import subprocess
import sys


DEP_FIELDS = (
    "--depends_fetch",
    "--depends_extract",
    "--depends_build",
    "--depends_lib",
    "--depends_run",
)


def port_deps(subport):
    """Return the set of ports `subport` depends on (resolved port names)."""
    proc = subprocess.run(
        ["port", "info", *DEP_FIELDS, subport],
        capture_output=True,
        text=True,
    )
    # `port info` exits non-zero if a field is empty for some versions; the
    # useful output is still on stdout, so parse regardless of return code.
    deps = set()
    for line in proc.stdout.splitlines():
        # Lines look like: "depends_lib: libcxx, clang-22, llvm-22"
        if ":" not in line:
            continue
        _, _, rhs = line.partition(":")
        for name in rhs.split(","):
            name = name.strip()
            if name:
                deps.add(name)
    return deps


def compute_waves(subports):
    """Topologically sort `subports` into dependency-ordered waves.

    Only edges between two changed subports are considered. Raises ValueError
    if a dependency cycle is detected among the changed subports.
    """
    changed = set(subports)
    # edge: port -> set of changed deps it must wait for
    blockers = {p: (port_deps(p) & changed) - {p} for p in subports}

    waves = []
    placed = set()
    remaining = set(subports)
    while remaining:
        ready = sorted(p for p in remaining if blockers[p] <= placed)
        if not ready:
            raise ValueError(
                f"dependency cycle among changed subports: {sorted(remaining)}"
            )
        waves.append(ready)
        placed.update(ready)
        remaining.difference_update(ready)
    return waves


def collapse_waves(waves, max_waves):
    """Collapse waves beyond max_waves into the last wave, keeping order.

    The collapsed wave lists ports in topological order, so chunking it keeps
    most dependencies either in earlier waves or earlier in the same chunk;
    anything else is rebuilt from source by install-dependencies.
    """
    if len(waves) <= max_waves:
        return waves
    collapsed = waves[: max_waves - 1]
    tail = [p for wave in waves[max_waves - 1 :] for p in wave]
    return collapsed + [tail]


def chunk_waves(waves, chunk_size):
    """Split each wave into labelled chunks of at most chunk_size subports."""
    result = []
    for w, wave in enumerate(waves):
        chunks = []
        for c in range(0, len(wave), chunk_size):
            chunks.append(
                {
                    "label": f"w{w}c{c // chunk_size}",
                    "ports": " ".join(wave[c : c + chunk_size]),
                }
            )
        result.append(chunks)
    return result


def wave_name(ports, max_len=48):
    """Human-readable wave name: as many port names as fit, then a count."""
    name = " ".join(ports)
    if len(name) <= max_len:
        return name
    shown = []
    length = 0
    for port in ports:
        if length + len(port) + 1 > max_len - 5:
            break
        shown.append(port)
        length += len(port) + 1
    # Always show at least one port, however long
    if not shown:
        shown = ports[:1]
    return " ".join(shown) + f" +{len(ports) - len(shown)}"


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--max-waves", type=int, default=12)
    parser.add_argument("subports", nargs="*")
    args = parser.parse_args(argv)

    subports = [p for p in args.subports if p.strip()]
    if not subports:
        waves = []
    else:
        waves = chunk_waves(
            collapse_waves(compute_waves(subports), args.max_waves),
            args.chunk_size,
        )

    names = [
        wave_name([p for chunk in wave for p in chunk["ports"].split()])
        for wave in waves
    ]

    print(json.dumps(waves))

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as fh:
            fh.write(f"waves={json.dumps(waves)}\n")
            fh.write(f"num_waves={len(waves)}\n")
            fh.write(f"wave_names={json.dumps(names)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
