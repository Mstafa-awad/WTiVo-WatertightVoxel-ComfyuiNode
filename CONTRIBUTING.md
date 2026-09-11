# Contributing to WTiVo

Contributions are welcome.

By submitting a contribution, you certify that you have the right to submit it and agree that WTiVo-authored contributions may be distributed under GPL-3.0-or-later. Changes to files already marked Apache-2.0 must preserve the Apache-2.0 header, upstream attribution, and prominent modification notice.

Before opening a pull request:

```text
python scripts/source_audit.py
```

On a configured Windows/NVIDIA machine also run:

```text
.venv\Scripts\python.exe scripts\verify_install.py
```

Geometry/quality changes should include an A/B log with input/final resolution, point budget, lambda_fill, final vertex/face counts, `bad_edge_groups`, hardware, and runtime. Performance-only changes must not silently alter graph capacities, label-query padding, contour resolution, or final topology behavior.
