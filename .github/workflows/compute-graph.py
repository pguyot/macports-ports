#!/usr/bin/env python3
"""Compute a flat, dependency-annotated build list for changed subports.

This is the "graph" alternative to compute-waves.py. Instead of grouping the
changed subports into a fixed chain of dependency waves (which needs a static
number of chained GitHub Actions jobs), it emits *one entry per subport* for a
single flat matrix, the way jmroot's dynamic-matrix prototype does -- but with
each entry annotated with the changed dependencies it must wait for.

Ordering between jobs is then enforced at run time, not by the job graph: a
job polls the workflow's artifacts for a `<dep>-built` / `<dep>-failed` marker
before building (see graph-build.yml). This lifts the wave design's two static
limits at once -- there is no maximum dependency depth, and a leaf never waits
on a barrier, only on its own dependencies.

Why one job per port and not one job per dependency *chain* (which would
amortize the ~5 min MacPorts bootstrap across a chain of dependent ports):
GitHub Actions makes an uploaded artifact visible to other running jobs only
once the uploading *step* finishes, and a job cannot emit a dynamic number of
steps. A chain job could therefore only publish its ports' archives when the
whole chain ends, which would make a dependent wait for an entire chain instead
of for the single port it needs -- defeating the pipelining. One job per port
is the granularity at which mid-run, per-port visibility is actually available.

Each emitted entry carries the *transitive* set of changed dependencies, not
just the direct ones, so a job restores the full PR-built closure of its
dependencies into ${portdbpath}/incoming/verified/ -- exactly what each
downstream wave does in the wave design.

Entries are emitted in topological order (dependencies before dependents) so
that, under GitHub's roughly FIFO matrix scheduling, the ports nearer the root
of the graph claim the limited macOS runners first and the graph fills in
bottom-up; this keeps dependents from holding runner slots while idly polling
for dependencies that have not been scheduled yet.

GitHub caps a single job's matrix at 256 configurations, so the entries are
also split into `--batch-size` batches: the workflow runs one matrixed build
job per batch (all depending only on `setup`, never on each other), which keeps
each matrix under the cap without imposing any dependency-depth limit -- the
run-time artifact ordering works across batches exactly as within one.

Usage:
    compute-graph.py [--batch-size N] SUBPORT [SUBPORT ...]

Emits the entry list as JSON on stdout:
    [{"port": "llvm-22", "waits": ""},
     {"port": "clang-22", "waits": "llvm-22"},
     {"port": "flang-22", "waits": "clang-22 llvm-22 mlir-22"}, ...]
and, when run inside GitHub Actions (GITHUB_OUTPUT set), writes `count`, the
batched entries as `batches` (a JSON list of entry lists) and `num_batches`.
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


def transitive_blockers(subports):
    """Map each subport to the changed subports it transitively depends on.

    Only edges between two changed subports matter: unchanged dependencies are
    installed as prebuilt binaries from the archive server and impose no
    ordering. Raises ValueError on a dependency cycle among changed subports.
    """
    changed = set(subports)
    direct = {p: (port_deps(p) & changed) - {p} for p in subports}

    # Topologically order the changed subgraph, then accumulate ancestors.
    order = []
    placed = set()
    remaining = set(subports)
    while remaining:
        ready = sorted(p for p in remaining if direct[p] <= placed)
        if not ready:
            raise ValueError(
                f"dependency cycle among changed subports: {sorted(remaining)}"
            )
        order.extend(ready)
        placed.update(ready)
        remaining.difference_update(ready)

    transitive = {}
    for p in order:  # dependencies precede dependents, so deps are ready
        acc = set(direct[p])
        for d in direct[p]:
            acc |= transitive[d]
        transitive[p] = acc
    return order, transitive


def main(argv):
    parser = argparse.ArgumentParser()
    # 250 leaves headroom under GitHub's hard cap of 256 matrix jobs per job.
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("subports", nargs="*")
    args = parser.parse_args(argv)

    subports = [p for p in args.subports if p.strip()]
    if not subports:
        entries = []
    else:
        order, transitive = transitive_blockers(subports)
        entries = [
            {"port": p, "waits": " ".join(sorted(transitive[p]))} for p in order
        ]

    # Topological order is preserved across batches (batch 0 holds the roots),
    # so GitHub's roughly FIFO scheduling still fills the graph bottom-up.
    batches = [
        entries[i : i + args.batch_size]
        for i in range(0, len(entries), args.batch_size)
    ]

    print(json.dumps(entries))

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as fh:
            fh.write(f"count={len(entries)}\n")
            fh.write(f"batches={json.dumps(batches)}\n")
            fh.write(f"num_batches={len(batches)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
