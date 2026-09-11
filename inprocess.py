# SPDX-License-Identifier: Apache-2.0
"""Low-overhead in-process WTiVo runner for ComfyUI.

This module deliberately keeps the proven WTiVo algorithm and bundled native
backend while removing the second Python/Torch process and temporary NPY bridge.
The public node exposes only practical controls; stable internal knobs are fixed
here to their tested values.
"""
from __future__ import annotations

import contextlib
import gc
import io
import logging
import time

import numpy as np

from . import wtivo as w


# Stable internals. These are intentionally not ComfyUI widgets.
GPU_LOCAL_STEPS = 8
GLOBAL_RELABEL_PERIOD = 1024
MAX_ROUNDS = 2_000_000
THICK_BAND_VOXELS = 3.0
THIN_BAND_VOXELS = 3.0
FAITHC_TRI_MODE = "auto"
FAITHC_CLAMP_ANCHORS = True
FAITHC_LAMBDA_N = 1.0
FAITHC_LAMBDA_D = 0.1


class _QuietOutput:
    """Suppress verbose Python/pybind prints while preserving a small error tail."""

    def __init__(self):
        self.buffer = io.StringIO()

    @contextlib.contextmanager
    def capture(self):
        with contextlib.redirect_stdout(self.buffer):
            yield

    def error_tail(self, lines: int = 24) -> str:
        text = self.buffer.getvalue().splitlines()
        return "\n".join(text[-lines:])


def _cleanup_host_and_cuda(full: bool = False) -> None:
    gc.collect()
    w.cleanup_cuda(full=full)
    w.compact_host_heap()


def process_arrays(
    vertices,
    faces,
    *,
    input_res: int = 1536,
    final_res: int = 1024,
    proxy_points: int = 12_000_000,
    proxy_eps_scale: float = 1.0,
    proxy_feature_weight: float = 1.5,
    lambda_fill: float = 20.0,
    threads: int = 16,
    thin_iso_vox: float = 0.0,
    faithc_component_mode: str = "auto",
):
    """Run WTiVo directly on Nx3/Mx3 arrays and return output arrays + audit info."""
    if faithc_component_mode not in ("auto", "keep_all", "largest"):
        raise ValueError("faithc_component_mode must be auto, keep_all, or largest")
    if float(proxy_feature_weight) < 0.0:
        raise ValueError("proxy_feature_weight must be >= 0")
    if int(threads) < 1:
        raise ValueError("threads must be >= 1")

    w.configure_resolutions(
        int(input_res), int(final_res), int(proxy_points), float(proxy_eps_scale)
    )

    # Validate once before entering the native-heavy path. Keeping the source as
    # direct arrays avoids the former temp-NPY files and second interpreter.
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"WTiVo vertices must be Nx3, got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"WTiVo faces must be Mx3, got {faces.shape}")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("WTiVo received an empty mesh")

    quiet = _QuietOutput()
    t_all = time.perf_counter()

    try:
        # 1) Sparse UDF -> exact point budget.
        t0 = time.perf_counter()
        with quiet.capture():
            thick_v, origin_min, origin_max, sparse_udf = w.direct_thick_points(
                None,
                float(proxy_feature_weight),
                int(threads),
                THICK_BAND_VOXELS,
                FAITHC_CLAMP_ANCHORS,
                FAITHC_LAMBDA_N,
                FAITHC_LAMBDA_D,
                input_vertices_array=vertices,
                input_faces_array=faces,
            )
        # The direct source arrays are no longer needed after the sparse field is built.
        vertices = None
        faces = None
        _cleanup_host_and_cuda(False)
        logging.info(
            "[WTiVo] Proxy: %s points | %.2fs",
            f"{len(thick_v):,}",
            time.perf_counter() - t0,
        )

        # 2) Same CGAL ThreadPack-v3 tetra path as the working CelloCut backend.
        t0 = time.perf_counter()
        thick_v64 = np.ascontiguousarray(thick_v, dtype=np.float64)
        del thick_v
        _cleanup_host_and_cuda(False)
        with quiet.capture():
            tet_verts, tets, neighbors = w.core.tetrahedralize_neighbors(
                thick_v64, int(threads)
            )
        del thick_v64
        _cleanup_host_and_cuda(True)
        logging.info(
            "[WTiVo] Tetra: %s cells | %.2fs",
            f"{len(tets):,}",
            time.perf_counter() - t0,
        )

        # 3) Move neighbors off host first, sample labels, then move tets off host.
        with quiet.capture():
            neighbors_cuda = w.topology_to_cuda_chunked(neighbors, "neighbors")
        del neighbors
        _cleanup_host_and_cuda(False)

        t0 = time.perf_counter()
        with quiet.capture():
            initial_labels, _label_seconds = sparse_udf.sample_tet_labels(
                np.ascontiguousarray(tet_verts, dtype=np.float64),
                np.ascontiguousarray(tets, dtype=np.int32),
                np.ascontiguousarray(np.asarray(origin_min, dtype=np.float64)),
                np.ascontiguousarray(np.asarray(origin_max, dtype=np.float64)),
                float(w.LABEL_QUERY_PADDING),
                float(w.LABEL_THRESHOLD),
                int(threads),
            )
        initial_labels = np.ascontiguousarray(
            np.asarray(initial_labels, dtype=np.uint8).reshape(-1)
        )
        del sparse_udf
        _cleanup_host_and_cuda(False)

        with quiet.capture():
            tets_cuda = w.topology_to_cuda_chunked(tets, "tets")
        del tets
        _cleanup_host_and_cuda(False)
        logging.info("[WTiVo] Labels/topology: %.2fs", time.perf_counter() - t0)

        # 4) WTiVo CUDA reduced graph / push-relabel.
        t0 = time.perf_counter()
        with quiet.capture():
            new_labels = w.gpupr_graph_cut_fast_v630(
                tet_verts,
                tets_cuda,
                neighbors_cuda,
                initial_labels,
                float(lambda_fill),
                int(threads),
                GLOBAL_RELABEL_PERIOD,
                MAX_ROUNDS,
                GPU_LOCAL_STEPS,
            )
        del initial_labels
        _cleanup_host_and_cuda(False)
        logging.info("[WTiVo] Graph cut: %.2fs", time.perf_counter() - t0)

        # 5) Extract the cut surface while topology remains GPU-resident.
        t0 = time.perf_counter()
        with quiet.capture():
            gc_v, gc_f = w.gpupr.surface_extraction_topology_cuda(
                new_labels, tet_verts, tets_cuda, neighbors_cuda, int(threads)
            )
        del new_labels, tet_verts, tets_cuda, neighbors_cuda
        _cleanup_host_and_cuda(True)
        logging.info(
            "[WTiVo] Surface: %s faces | %.2fs",
            f"{len(gc_f):,}",
            time.perf_counter() - t0,
        )

        # 6) Signed VDB -> direct FaithC contour + one final watertight audit.
        t0 = time.perf_counter()
        thin_payload = [gc_v, gc_f]
        del gc_v, gc_f
        with quiet.capture():
            final_v, final_f, watertight, bad_edges = w.direct_thin_mesh_owned(
                thin_payload,
                int(threads),
                float(thin_iso_vox),
                THIN_BAND_VOXELS,
                FAITHC_TRI_MODE,
                FAITHC_CLAMP_ANCHORS,
                FAITHC_LAMBDA_N,
                FAITHC_LAMBDA_D,
                str(faithc_component_mode),
            )
        _cleanup_host_and_cuda(True)
        logging.info("[WTiVo] Final contour: %.2fs", time.perf_counter() - t0)

        final_v = np.ascontiguousarray(final_v, dtype=np.float32)
        final_f = np.ascontiguousarray(final_f, dtype=np.int32)
        total = time.perf_counter() - t_all
        logging.info(
            "[WTiVo] Done: %s vertices / %s faces | watertight=%s | %.2fs",
            f"{len(final_v):,}",
            f"{len(final_f):,}",
            bool(watertight),
            total,
        )
        return final_v, final_f, bool(watertight), int(bad_edges), float(total)

    except BaseException:
        tail = quiet.error_tail()
        if tail:
            logging.error("[WTiVo] Suppressed diagnostic tail:\n%s", tail)
        raise
    finally:
        _cleanup_host_and_cuda(True)
