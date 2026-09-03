# Kaggle Run Plan for the CausalDEM-QEC Pilot

> **Purpose:** Run the existing 88-trajectory pilot configuration on a Kaggle
> Free CPU Notebook while preserving deterministic artifacts, private sealed
> inputs, and resumability.

## Decision

Kaggle is a useful second execution environment because its current CPU
Notebook specification provides about 30 GB RAM and 4 CPU cores, compared with
the local 16 GB WSL2 limit. Kaggle CPU sessions currently allow up to 12 hours
of execution and provide 20 GB of auto-saved `/kaggle/working` storage.

Sources: [Kaggle Notebook technical specifications](https://www.kaggle.com/docs/notebooks)
and [Kaggle private dataset quota announcement](https://www.kaggle.com/product-announcements/512322).

Use the CPU runtime. A GPU is not expected to accelerate the Stim/PyMatching
generation path and would waste limited accelerator quota.

This plan uses the unchanged scientific configuration
`configs/poc_pilot.json`: 4,096 burn-in rounds, 8,192 scored rounds, 32-round
episodes, 256-round blocks, the four circuits, and the approved 88-job pilot
matrix. It does not use the full-production `configs/poc.json`.

## Important limitation

The current repository can generate artifacts, but it is not yet a reliable
single-session Kaggle workflow:

- `generate-pilot` requires an 80 GiB storage reserve, which Kaggle's
  `/kaggle/working` does not provide.
- Kaggle scratch storage disappears when the session ends.
- A normal `generate-pilot` invocation schedules all remaining jobs, so a
  notebook interruption can occur before a checkpoint is uploaded.

Therefore, this plan requires a small additive Kaggle/checkpoint feature before
the operational run. It must not change `configs/poc_pilot.json`, the scientific
round geometry, seed derivation, artifact schema, or the existing default
generation mode.

## Required additive feature

Add a Kaggle execution mode to the existing CLI and generation code. Keep the
standard path unchanged and opt in explicitly:

```text
--execution-backend kaggle
--job-limit 1
--checkpoint-root /kaggle/working/checkpoint
--checkpoint-identity owner/checkpoint-dataset@exact-version
```

The implementation should:

1. Validate the exact pilot configuration and preserve its resolved-config
   hash.
2. Permit a small output root without applying the local 80 GiB preflight.
3. Generate one job, or a bounded small number of jobs, per notebook cell/run.
4. Update the manifest only after a complete observable/label pair verifies.
5. Export a checkpoint containing the manifest and completed artifact pairs.
6. Resume only from verified matching artifacts.
7. Reject execution-backend, code-version, configuration, or checkpoint
   mismatches rather than overwriting data.
8. Keep the sealed manifest out of every public or shared checkpoint.

The Kaggle mode may use the bounded-memory generation mode described in
`ram_bottleneck_solution.md`, but it must remain separately provenance-bound.
The first Kaggle attempt should use one worker and one trajectory per
checkpoint. If the 30 GB RAM environment completes a surface trajectory safely
with the standard sampler, the code may retain the standard scientific path;
otherwise use the opt-in bounded surface backend.

## Storage model

Use three separate locations:

```text
/kaggle/input/causaldem-pilot-checkpoint/   read-only previous checkpoint
/kaggle/working/runs/pilot/                 writable current job/output
/kaggle/working/export/                     files to upload as next checkpoint
```

Never write into `/kaggle/input`. Copy only the verified checkpoint contents
needed by the current run into the writable working directory. The copy must
include the existing `run_manifest.json` and both artifact lanes for all 44
completed jobs.

If the writable output approaches 18 GB, stop after the current complete job,
verify it, and publish a new checkpoint. Do not wait for the 20 GB limit. If the
complete pilot is larger than one Kaggle Dataset, shard checkpoints by circuit
family or pilot partition and record the shard list in a private manifest.

Before starting, check the account's private dataset quota in Kaggle. The quota
is account-dependent even though Kaggle has announced a 200 GB private dataset
allocation; do not assume that the whole amount is available.

## Private-data rules

The sealed manifest is an operational secret. It must be provided through a
Kaggle Secret or a private, access-controlled input and written only to a
session-local path such as:

```text
/kaggle/working/secrets/causaldem_pilot_sealed.json
```

Do not put it in:

- a public Kaggle Dataset;
- the notebook source;
- notebook output metadata;
- a committed repository directory;
- a checkpoint uploaded for collaborators who are not authorized for sealed
  evaluation.

The notebook must verify the sealed commitment before scheduling sealed jobs.
The existing code's explicit `--sealed-manifest` path and purpose checks remain
the source of truth.

## Prepare the private Kaggle inputs

From the local production `main` worktree, create a source archive containing
only the public implementation checkout. Do not archive the primary planning
checkout, `.superpowers`, private design files, sealed manifests, or the local
`.worktrees` directory itself.

Upload two private Kaggle Datasets:

1. `causaldem-poc-source`: the exact public-main source commit, `pyproject.toml`,
   `uv.lock`, `configs/`, `src/`, and tests. Include a text file named
   `COMMIT_SHA.txt` containing the exact public-main commit SHA.
2. `causaldem-pilot-checkpoint-00`: the verified current `runs/pilot` root with
   44 complete pairs and its manifest.

Record the source commit SHA and immutable checkpoint dataset version in the
notebook metadata and in the checkpoint manifest. Set
`CAUSALDEM_CHECKPOINT_VERSION` to the exact pinned input version in
`owner/dataset@number` form (or to a verified `sha256:<digest>` identity) and
pass it as `--checkpoint-identity`. Do not use a staging path or mutable latest
pointer, and do not regenerate the 44 pairs.

## Kaggle notebook setup

Create a private CPU notebook and attach the two private datasets. Enable
Internet only for dependency installation and checkpoint upload.

The first cells should verify the runtime before doing any generation:

```python
import platform
import shutil
import sys

print(sys.version)
print(platform.platform())
print(shutil.disk_usage("/kaggle/working"))
```

Then install and verify the exact locked environment:

```python
!python -m pip install -q uv
!uv sync --frozen --extra dev
!uv run python - <<'PY'
import importlib.metadata
for name in ("numpy", "scipy", "stim", "pymatching", "pyarrow"):
    print(name, importlib.metadata.version(name))
PY
```

If Python is not compatible with the repository's declared Python 3.11
environment, stop. Do not silently upgrade dependencies or regenerate the
lockfile inside Kaggle.

Copy the checkpoint into a writable root and verify it before generation:

```python
from pathlib import Path
import shutil

source_checkpoint = Path("/kaggle/input/causaldem-pilot-checkpoint-00/pilot")
working_root = Path("/kaggle/working/runs/pilot")
if working_root.exists():
    raise RuntimeError("working checkpoint already exists; inspect before reuse")
shutil.copytree(source_checkpoint, working_root)
```

Run the repository's dataset verification command before adding any new jobs.
The command must report the existing 44 verified pairs and must not report a
configuration or pair-identity mismatch.

## Checkpoint loop

The notebook must use a one-job checkpoint loop rather than one long generation
call:

```text
load latest verified checkpoint
        |
select the next sorted incomplete pilot job
        |
generate exactly one trajectory with workers=1
        |
verify both artifact lanes and update run_manifest.json
        |
copy manifest + verified artifact pair into export staging
        |
publish a new private Kaggle Dataset version
        |
stop the session or continue only if RAM, disk, and time budgets are safe
```

The next job is selected by the existing deterministic job ordering. A
checkpoint is valid only when:

- the manifest resolved-config hash matches `configs/poc_pilot.json`;
- every completed result has both matching artifact lanes;
- every artifact checksum and pair ID verifies;
- no staging directory is included as completed data;
- the checkpoint records the exact source commit and execution backend;
- the sealed commitment hash matches when sealed jobs are present.

Upload only after the current pair is complete. If upload fails, retain the
local working checkpoint and retry the upload; never regenerate the pair merely
because publishing failed.

## Checkpoint publishing

Use the Kaggle CLI/API from the notebook with a private dataset target. The
credential must come from Kaggle's secret mechanism and must not be printed.
The exact dataset slug is created once and then versioned:

```bash
if [ ! -f /kaggle/working/export/dataset-metadata.json ]; then
  kaggle datasets init -p /kaggle/working/export
fi
checkpoint_count="$(python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("/kaggle/working/runs/pilot/run_manifest.json").read_text())
print(sum(item.get("completed") is True for item in manifest["results"]))
PY
)"
source_commit="$(cat /kaggle/input/causaldem-poc-source/COMMIT_SHA.txt)"
kaggle datasets version \
  -p /kaggle/working/export \
  -m "pilot checkpoint: completed ${checkpoint_count} of 88; source ${source_commit}" \
  --dir-mode zip
```

The export directory must contain only the checkpoint root, manifest, verified
artifact files, and the manifest-bound public
`data/manifests/sealed_commitment.json` needed for sealed resume validation.
Exclude the private sealed manifest, source code, secrets, notebook outputs,
core dumps, logs containing seeds, and scratch directories.

After publishing, record the returned dataset version or creation timestamp in
a local notebook log. On the next session, attach that exact version rather
than relying on a mutable latest pointer.

## Recommended run schedule

### Session 0: validation

- Attach source and checkpoint datasets.
- Verify Python, package versions, disk, and RAM.
- Verify all 44 existing artifact pairs.
- Run one nonsealed surface job in a temporary root.
- Measure peak RSS and wall time.
- Do not touch `runs/pilot` until the trajectory passes all checks.

### Session 1 onward: generation

- Attach the newest private checkpoint.
- Copy it into `/kaggle/working/runs/pilot`.
- Generate one sorted incomplete job at a time.
- Publish a private checkpoint after every successful job.
- Stop before the 12-hour limit or 18 GB working-storage threshold.

Prioritize surface jobs first because the local failure occurred before any
surface artifact was published. Keep `workers=1` throughout.

## Code implementation checklist

Implement this as a separately reviewed additive change in the existing public
worktree:

1. In `core.py`, add an immutable execution-options value containing backend,
   job limit, checkpoint identity, and optional bounded-generation settings.
2. In `cli.py`, add `--execution-backend`, `--job-limit`, and
   `--checkpoint-root`; reject invalid combinations before workers start.
3. In `simulate.py`, add deterministic next-job selection and a one-job
   generation entry point that reuses `generate_matrix` transactions and
   existing artifact verification. Do not duplicate scientific generation
   logic in the CLI.
4. In `artifacts.py`, add checkpoint inventory/export helpers that copy only
   verified final artifact directories and the parent manifest. Never copy
   staging directories or private sealed files.
5. Bind source commit, execution backend, generation-law version, and checkpoint
   identity in the run manifest without changing the scientific config hash.
6. Add tests in `tests/test_core_artifacts.py`, `tests/test_simulation.py`, and
   `tests/test_pipeline.py` for one-job limits, checkpoint inventory, resume,
   unchanged existing hashes, private-seed rejection, and clean/resumed
   determinism.
7. If the standard surface path still exceeds 30 GB, implement and invoke the
   opt-in bounded surface backend from `ram_bottleneck_solution.md`; keep
   standard mode unchanged.

Each implementation task must use the repository's required sequential
subagent-driven development, independent specification review, focused tests,
and explicit public-worktree commits. Do not push as part of Kaggle setup.

### Final session: verification and export

- Attach the checkpoint containing all 88 pairs.
- Run `verify-dataset` against the exact pilot configuration.
- Run the required EDA/report stages if their output fits the working disk.
- Export reports separately if needed; do not mix report files into the
  artifact checkpoint.
- Download the final private checkpoint to the local production worktree.
- Run the complete local verification suite before treating the pilot as
  complete.

## Failure handling

- Session timeout: resume from the last published checkpoint.
- OOM: preserve the last checkpoint, reduce only the bounded execution chunk
  size, and record the new generation-law identity before retrying.
- Disk threshold: stop, publish, and start a new session from the checkpoint.
- Dependency mismatch: stop and rebuild the notebook environment from `uv.lock`.
- Checksum or pair conflict: stop; never overwrite the existing artifact.
- Sealed commitment mismatch: stop; do not regenerate or freeze a new sealed
  manifest.
- Kaggle upload failure: retry upload from the same verified export directory.
- Unexpected Python/native crash: mark the current job incomplete and rerun it
  only after checking that no complete pair was published.

## Acceptance criteria

The Kaggle path is accepted only when all of the following are true:

- the current `configs/poc_pilot.json` is used unchanged;
- all 88 expected jobs are present exactly once;
- all 88 observable/label pairs verify;
- existing 44 pair hashes remain unchanged;
- clean and resumed Kaggle runs produce identical new surface hashes;
- standard local smoke behavior remains unchanged;
- worker-independent deterministic tests pass where the selected backend
  supports multiple workers;
- all applicable DQ gates pass, with DQ08 handled according to the existing
  target-selection stage;
- no raw sealed seed appears in any artifact, checkpoint, notebook output, or
  log;
- the final downloaded root passes local `verify-dataset`.

## Local final verification

Run from `.worktrees/public-main` after downloading the final checkpoint:

```bash
uv sync --frozen --extra dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run pytest -m slow -q
uv run causaldem-poc verify-dataset \
  --config configs/poc_pilot.json \
  --output-root runs/pilot
```

Then run the notebook/report export required by the approved plan. Reports must
retain `PILOT / NOT FINAL` and must not elevate the pilot to full-production
evidence.

## Explicit stop conditions

Stop the Kaggle plan if:

- the account cannot store the private checkpoint shards;
- the output cannot be checkpointed before the 20 GB working limit;
- the 30 GB RAM CPU runtime still OOMs on one surface trajectory;
- the exact locked dependencies cannot be installed;
- the sealed commitment cannot be kept private and verifiable;
- a checkpoint would require rewriting or deleting an existing artifact.

In any of these cases, use the bounded-memory implementation on a machine with
more persistent disk, or provision a different Linux host. Do not weaken the
pilot configuration to make Kaggle fit.
