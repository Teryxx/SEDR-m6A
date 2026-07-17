# Dataset Label History Rewrite Design

## Goal

Replace experiment-style checkpoint labels with dataset names and remove the
manifest `code` field from every commit reachable from `main`. The rewrite must
leave model behavior, datasets, checkpoint weights, and inference inputs
unchanged.

## Scope

- Rewrite the complete history reachable from `main` (six commits at audit
  time, plus this design commit).
- Update the 11 JSON files under `checkpoint/configs/`.
- Remove the `code` property or column from historical JSON and CSV manifests.
- Preserve all unrelated metadata and all model, data, and LFS content.
- Update only `main`; the repository has no other branches or tags.

## Transformation Rules

For each `checkpoint/configs/<dataset>.json` in each historical tree:

1. Derive the dataset name from the config filename without its extension.
2. Read the config's original `grid_code` value as the label being replaced.
3. Set `grid_code` to the dataset name.
4. Replace that original label in `output_dir` and `run_name` with the dataset
   name.
5. In `source_config_path`, replace any known original checkpoint label with
   the dataset name of the config that owns the field. This handles a source
   path that contains a label selected by another dataset without creating a
   misleading cross-dataset name.
6. Leave `grid_label`, hyperparameters, checkpoint paths, and all other values
   unchanged.

For `checkpoint/final11_manifest.json` and
`checkpoint/final11_manifest.csv` in each historical tree:

1. Remove the `code` property or column when present.
2. Preserve row order and every remaining value.

The rewrite must abort before pushing if an original label remains outside
these explicitly handled locations.

## History Rewrite

Use a disposable full clone and a path-aware history filter. Preserve commit
order, parent relationships, author identity, author dates, committer identity,
committer dates, and commit messages. Commit hashes will necessarily change.

After approval, push this design commit normally so it is part of the history
being rewritten. Then record the remote `main` object ID and create the
disposable full clone. Push the rewritten branch with an explicit
`--force-with-lease` expectation for the recorded object ID. Do not create a
remote backup branch or tag because it would keep the old history reachable.
If the remote branch moves, stop without pushing.

GitHub may retain unreachable objects internally for some time after a force
push. The success criterion is that the old content is absent from every commit
reachable from the repository's advertised refs.

## Validation

Before the force push:

- Parse every rewritten config and both manifest formats.
- Confirm every `grid_code` equals its config filename's dataset name.
- Confirm all 11 original labels have zero matches across all reachable commits.
- Confirm historical manifests have no `code` property or column.
- Confirm `grid_label` and all unrelated current config values match the
  pre-rewrite tree.
- Confirm current checkpoint and data blob IDs, including LFS pointer blobs,
  match the pre-rewrite tree.
- Confirm the current JSON and CSV manifests still contain the same 11 records
  and remaining values.

After the force push, create a fresh clone with LFS downloads disabled and
repeat the all-history label scan, manifest checks, config checks, and remote
head comparison.

## Failure Handling

- Stop before pushing on any parsing, comparison, or scan failure.
- Stop if `main` no longer points to the recorded pre-rewrite object ID.
- If remote verification fails after pushing, restore the recorded old object
  ID with another explicit lease while the rewrite clone is still available,
  then report the failure.
