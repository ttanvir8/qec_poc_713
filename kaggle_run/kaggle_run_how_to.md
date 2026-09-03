# Kaggle Pilot Run How-To

This guide runs the CausalDEM-QEC pilot on the Kaggle free CPU tier. It uses the
existing package CLI and writes only private checkpoint artifacts. Do not paste
actual secrets or raw seed values into the notebook, this guide, Git, notebook
metadata, or any Kaggle Dataset output.

## 1. Prepare private local inputs

1. Work from the public implementation checkout on `main`, not the private
   planning checkout.
2. Confirm the public source commit:

   ```bash
   git -C /mnt/d/win-fun/quantum/poc_713/.worktrees/public-main rev-parse HEAD
   git -C /mnt/d/win-fun/quantum/poc_713/.worktrees/public-main status --short
   ```

3. Create a source upload directory that contains only publishable source files.
   Include `pyproject.toml`, `uv.lock`, `configs/`, `src/`, `tests/`,
   `kaggle_run/`, and a `COMMIT_SHA.txt` file containing the exact public
   source commit.
4. Exclude `.superpowers/`, the primary checkout, `.worktrees/`, private
   planning documents, sealed manifests, generated `runs/`, generated `data/`,
   generated `reports/`, shell history, and notebook outputs.
5. Upload that directory as a private source dataset named
   `causaldem-poc-source`.
6. Upload the verified current pilot checkpoint as a private checkpoint dataset,
   for example `causaldem-pilot-checkpoint-00`. Its root should contain
   `pilot/run_manifest.json` and the matching observable and label artifact
   lanes. Do not regenerate the existing 44-pair checkpoint.

## 2. Create the private Kaggle notebook

1. Create a new private Notebook.
2. Select the Kaggle free CPU runtime. Do not select a GPU.
3. Attach the private source dataset `causaldem-poc-source`.
4. Attach the newest private checkpoint dataset, initially the 44-pair
   checkpoint dataset.
5. Enable Internet only while installing dependencies or publishing a checkpoint
   dataset version.
6. Add Kaggle Secrets named `KAGGLE_USERNAME` and `KAGGLE_KEY` for the Kaggle
   API. For sealed jobs only, add `CAUSALDEM_PILOT_SEALED_MANIFEST` containing
   the sealed manifest. Do not paste the sealed manifest into a cell.
7. Set notebook environment variables before publishing:

   ```python
   import os

   os.environ["CAUSALDEM_CHECKPOINT_DATASET_SLUG"] = "your-kaggle-name/causaldem-pilot-checkpoint"
   os.environ["CAUSALDEM_CREATE_CHECKPOINT_DATASET"] = "1"  # first upload only
   ```

   For later versions, remove `CAUSALDEM_CREATE_CHECKPOINT_DATASET` or set it to
   `0`.

## 3. Run the notebook asset

1. Open `kaggle_run/kaggle_pilot_runner.py` from the private source dataset.
2. Paste it into the Kaggle notebook as cells, or run it as a script from the
   copied source directory.
3. The first cells verify the runtime and print Python, platform, disk, RAM, and
   elapsed-time checks. Stop if Python is not 3.11, if working storage is near
   18 GiB used, if less than 2 GiB is free, or if the notebook is too close to
   the 12-hour session limit.
4. The setup installs and verifies the frozen dependency set:

   ```bash
   uv sync --frozen --extra dev
   ```

   Do not regenerate `uv.lock` in Kaggle.
5. The source commit check compares the attached private source dataset's
   `COMMIT_SHA.txt` to the copied source tree and, when `.git` is present, to
   `git rev-parse HEAD`.
6. The checkpoint copy step copies the private checkpoint dataset into
   `/kaggle/working/runs/pilot`. The notebook never writes into
   `/kaggle/input`.

## 4. Legacy bootstrap manifest upgrade

The initial 44-pair checkpoint may be a legacy bootstrap manifest without
Kaggle provenance. The notebook calls:

```python
upgrade_legacy_kaggle_bootstrap(...)
```

The upgrade is accepted only for the known 44-pair checkpoint shape and only
when artifact lanes, resolved config hash, expected job keys, sealed commitment
hash, and checkpoint provenance all match. The upgrade writes provenance into
`/kaggle/working/runs/pilot/run_manifest.json`; it does not expose a sealed
manifest or any raw seed values.

If the manifest is already provenance-bound, the notebook verifies it with the
checkpoint inventory API and continues. If the legacy upgrade fails, stop and
inspect the attached checkpoint dataset version instead of editing the manifest
by hand.

## 5. Verify before generation

Run the repository verifier against the copied checkpoint before scheduling new
work:

```bash
uv run causaldem-poc verify-dataset \
  --config configs/poc_pilot.json \
  --output-root /kaggle/working/runs/pilot
```

For the bootstrap checkpoint, expect the existing completed pairs to verify.
Configuration, pair identity, checksum, lane, provenance, or sealed commitment
mismatches are stop conditions.

## 6. One-job CLI loop

Run exactly one pilot job per checkpoint loop:

```bash
uv run causaldem-poc generate-pilot \
  --config configs/poc_pilot.json \
  --output-root /kaggle/working/runs/pilot \
  --workers 1 \
  --execution-backend kaggle \
  --job-limit 1 \
  --checkpoint-root /kaggle/working/export/pilot
```

For sealed jobs, the runner adds:

```bash
--sealed-manifest /kaggle/working/secrets/causaldem_pilot_sealed.json
```

That file is created from a Kaggle Secret only, under private permissions, and
is never copied into the checkpoint export. Keep `--workers 1` on the Kaggle
free CPU tier.

## 7. Verified export staging

After a successful job, the CLI exports a checkpoint to
`/kaggle/working/export/pilot`. The export is uploadable only if it contains the
manifest and verified artifact files, and excludes source code, secrets,
staging directories, logs containing sensitive values, `.superpowers/`,
`.worktrees/`, notebook outputs, and scratch files.

The runner checks the export root before upload. If generation completed but no
export exists, stop and inspect the CLI output; do not publish a partial
checkpoint.

## 8. Private Kaggle Dataset versioning

The first successful export creates the private checkpoint dataset:

```bash
kaggle datasets create -p /kaggle/working/export --private --dir-mode zip
```

Every subsequent export versions the same private Kaggle Dataset:

```bash
kaggle datasets version \
  -p /kaggle/working/export \
  -m "pilot checkpoint: completed <count> of 88; source <source commit>" \
  --dir-mode zip
```

Record the returned Kaggle Dataset version or timestamp in local run notes. In
the next Kaggle session, attach that exact private checkpoint dataset version
instead of relying on a mutable latest pointer.

## 9. Storage, time, and retry rules

Use these thresholds on Kaggle free CPU:

- Stop after the current complete job if `/kaggle/working` approaches 18 GiB
  used.
- Stop if less than 2 GiB of `/kaggle/working` is free.
- Stop before the 12-hour limit; leave at least 45 minutes for verification and
  upload.
- If generation fails before a pair is complete, resume from the last published
  private checkpoint dataset version.
- If upload fails, retry the upload from the same verified
  `/kaggle/working/export` directory. Do not regenerate the just-completed pair
  merely because publishing failed.
- If source commit, config hash, execution backend, checkpoint identity,
  artifact checksum, pair ID, or sealed commitment checks fail, stop and attach
  the correct dataset version.

## 10. Sealed manifest handling

The sealed manifest must be supplied via Kaggle Secret only. Do not paste it
into notebook source, notebook output, a committed file, a public Kaggle
Dataset, or a shared checkpoint dataset. The checkpoint may contain only the
sealed commitment hash recorded by the current artifact APIs.

Set `CAUSALDEM_USE_SEALED_MANIFEST=1` only for sessions expected to schedule
sealed jobs. Leave it unset for normal and development jobs.

## 11. Final local verification

This is the final local verification step before treating the private pilot
checkpoint as ready for downstream pilot reports.

After the checkpoint containing all 88 pairs is published, download that private
Kaggle Dataset version into the local public worktree and run:

```bash
uv run causaldem-poc verify-dataset \
  --config configs/poc_pilot.json \
  --output-root runs/pilot
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run pytest -m slow -q
```

Then run the report or notebook export stages required by the project plan. Keep
reports separate from checkpoint artifacts and label all pilot outputs
`PILOT / NOT FINAL`.
