# Licensing and commercial use

WTiVo is intended to be **free and open source, including for commercial use**.
The public open-source build uses CGAL's GPL-licensed 3D Triangulations package, so the combined WTiVo distribution is released under **GNU GPL v3 or later**.

This is a practical project-maintainer summary, not legal advice.

## What the GPL allows

You may use WTiVo personally, academically, internally at a company, or commercially. You may charge for services performed with it and you may sell copies. The GPL does **not** prohibit commercial use.

The main obligation appears when you **distribute** WTiVo or a modified binary: recipients must receive the GPL freedoms and corresponding source as required by GPLv3. Private/internal modifications do not automatically have to be published merely because you run them internally.

WTiVo does not claim ownership of a user's input meshes or generated output meshes.

## Why the repository is not MIT-only

CelloCut and FaithC are Apache-2.0 projects and most of WTiVo's sparse/native reconstruction code can coexist with permissive licenses. However WTiVo uses CGAL 3D Delaunay triangulation. CGAL documents its 3D Triangulations package as GPL and provides a separate commercial-license option. Under the free CGAL license, software distributed based on those GPL data structures must comply with the GPL.

Apache-2.0 code is compatible with GPLv3, so Apache-derived files can be distributed in this GPLv3 combined program while retaining their Apache notices and terms.

## File-level licenses

- `LICENSE` — GPL-3.0-or-later for the combined WTiVo distribution.
- `wtivo.py` — Apache-2.0 because it is a modified/derived production runner built from the Apache-2.0 CelloCut lineage.
- `native/core/wtivo_core.cpp` — Apache-2.0 source modifications, compiled against GPL CGAL 3D Triangulations.
- `native/vdb/wtivo_vdb.cpp` — Apache-2.0; modified WTiVo sparse OpenVDB/FaithC-compatible bridge.
- `native/gpupr/*` — Apache-2.0; modified WTiVo graph-cut/CUDA implementation.
- WTiVo-authored setup/build/test/docs files — GPL-3.0-or-later unless otherwise marked.

SPDX headers are authoritative for individual source files. The combined open-source program remains GPL-3.0-or-later because of CGAL.

## Closed-source/proprietary redistribution

Do **not** assume this open-source build can be embedded into a closed-source product. If you need proprietary redistribution, obtain appropriate commercial rights for CGAL and independently review the remaining licenses and your distribution architecture. WTiVo does not sell or grant a CGAL commercial license.

## CUDA / NVIDIA

WTiVo source does not bundle the NVIDIA CUDA Toolkit, driver, or PyTorch binaries. The Windows installer directs users to/install-from the official providers. Those external components remain governed by their own terms. The source repository itself is not a redistribution of CUDA.

## Authoritative upstream references

- CelloCut: https://github.com/rangeryx-66/CelloCut
- FaithC: https://github.com/Luo-Yihao/FaithC
- CGAL license: https://www.cgal.org/license.html
- CGAL 3D Triangulations: https://doc.cgal.org/latest/Triangulation_3/
- OpenVDB: https://github.com/AcademySoftwareFoundation/openvdb
- oneTBB: https://github.com/uxlfoundation/oneTBB
- Eigen: https://gitlab.com/libeigen/eigen
- PyTorch: https://github.com/pytorch/pytorch
- NumPy: https://github.com/numpy/numpy
- Trimesh: https://github.com/mikedh/trimesh
- pybind11: https://github.com/pybind/pybind11
- vcpkg: https://github.com/microsoft/vcpkg
