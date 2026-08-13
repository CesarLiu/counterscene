# Contributing to CounterScene

Thank you for improving CounterScene. Please open an issue before a large change
so its scope and compatibility requirements can be discussed.

## Development checks

Before submitting a pull request, run:

```bash
python -m unittest discover -s tests -v
bash -n scripts/*.sh
```

Keep CounterScene-specific logic in `ccdiff/counterscene` where possible and
avoid duplicating implementations already provided by the vendored tbsim
runtime. New behavior should include a focused test and documentation for any
new public configuration or artifact field.

## Release boundaries

Do not contribute:

- vehicle-selection implementation or private selection intermediates;
- nuScenes data, model checkpoints without redistribution permission, or raw
  evaluation outputs;
- credentials, machine-specific absolute paths, or personal information.

Changes under `third_party/tbsim` remain subject to the NVIDIA Source Code
License-NC. Other contributions are submitted under the repository's
Apache-2.0 license unless a file states otherwise. Contributors must have the
right to submit their work under the applicable license.
