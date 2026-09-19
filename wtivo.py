# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Mstafa-awad
#
# WTiVo: WatertightVoxel Optimizer
# Modified for WTiVo in 2026. Production runner derived from the Apache-2.0
# CelloCut method and paired
# with Apache-2.0 FaithC-inspired contouring/native components.
# See LICENSE, NOTICE and THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import argparse
import ctypes
import os
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Mirror standalone CLI output to wtivo_execution.log.
# Skipped when:
#  - imported as a module (ComfyUI in-process),
#  - running as our subprocess worker (WTIVO_SUBPROCESS),
#  - attached to an interactive console (dup2 onto a console handle
#    causes OSError 9 / "lost sys.stderr" on Windows).
if (
    os.name == "nt"
    and __name__ == "__main__"
    and not os.environ.get("WTIVO_SUBPROCESS")
    and not sys.stdout.isatty()
):
    try:
        _log_path = ROOT / "wtivo_execution.log"
        _log_file = open(_log_path, "w", encoding="utf-8", buffering=1)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(_log_file.fileno(), 1)
        os.dup2(_log_file.fileno(), 2)
    except Exception as e:
        print(f"[WTiVo-Fix] Stream redirection warning: {e}", flush=True)
        
for _p in (
    ROOT,
    ROOT / "build",
):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Windows DLL lookup.
# WTiVo ships its own third-party native runtime DLLs in build/.
# PyTorch/CUDA DLLs come from the user's ComfyUI installation.
# No dependency on any CelloCut/Conda environment.
_DLL_HANDLES = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    py = Path(sys.executable).resolve().parent
    torch_lib = py / "Lib" / "site-packages" / "torch" / "lib"
    candidates = [
        ROOT / "build",
        py,
        torch_lib,
        ROOT / ".deps" / "vcpkg" / "installed" / "x64-windows" / "bin",
    ]
    for key in ("CUDA_PATH", "CUDA_HOME"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value) / "bin")
    for d in candidates:
        try:
            if d.exists():
                _DLL_HANDLES.append(os.add_dll_directory(str(d)))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        except OSError:
            pass

import numpy as np
import torch
import trimesh

# Prefer a native WTiVo build when present.  This no-build package also ships
# the already-working CelloCut Python-3.12/Windows-x64 binaries supplied by the
# user.  The adapter below exposes WTiVo's small native API without importing
# CelloCut.py or its ComfyUI node logic.
_BACKEND_NAME = "native-wtivo"
try:
    import wtivo_core as core
    import wtivo_gpupr as gpupr
    import wtivo_vdb as vdb
except (ImportError, OSError) as _wtivo_native_error:
    try:
        import cppmodules as _legacy_cpp
        import cellocut_gpupr_fast as _legacy_gpupr
        import cellocut_vdb_watertight as vdb
    except (ImportError, OSError) as _legacy_error:
        raise ImportError(
            "WTiVo could not load either a native WTiVo build or the bundled "
            "CelloCut-compatible prebuilt backend. This package requires Windows x64, "
            "Python 3.12, the same Torch/CUDA runtime used by the working CelloCut node, "
            "and its TBB/GMP runtime DLLs. Native error: "
            f"{_wtivo_native_error!r}; prebuilt error: {_legacy_error!r}"
        ) from _legacy_error

    class _LegacyCoreAdapter:
        @staticmethod
        def tetrahedralize_neighbors(vertices, threads=0):
            # CelloCut ThreadPack-v3 handles its own parallelism.
            return _legacy_cpp.tetrahedralize_neighbors_v3(vertices)

        @staticmethod
        def largest_component(vertices, faces, threads=0):
            return _legacy_cpp.largest_component_parallel_v3(vertices, faces, int(threads))

        @staticmethod
        def is_watertight(faces, threads=0):
            return _legacy_cpp.is_watertight_edges_v52(faces, int(threads))

    class _LegacyGpuAdapter:
        @staticmethod
        def graph_cut_fast(vertices, tets_cuda, neighbors_cuda, labels, fill_T, threads=16,
                           global_relabel_period=1024, max_rounds=2000000, local_steps=8):
            return _legacy_gpupr.graph_cut_gpupr_fast_v630(
                vertices, tets_cuda, neighbors_cuda, labels, float(fill_T), int(threads),
                int(global_relabel_period), int(max_rounds), int(local_steps)
            )

        @staticmethod
        def surface_extraction_topology_cuda(labels, vertices, tets_cuda, neighbors_cuda, threads=16):
            return _legacy_cpp.surface_extraction_topology_cuda_v56(
                labels, vertices, tets_cuda, neighbors_cuda, int(threads)
            )

    core = _LegacyCoreAdapter()
    gpupr = _LegacyGpuAdapter()
    _BACKEND_NAME = "prebuilt-cellocut-compatible"

print(f"[WTiVo] native backend: {_BACKEND_NAME}", flush=True)

# Runtime-configurable resolutions. Defaults reproduce the tested WTiVo benchmark.
R = 1536
FINAL_R = 1536
PROXY_POINTS = 12_000_000
PROXY_EPS_SCALE = 1.0

EPS = 1.0 / R
BAND = 3.0 / R
VOXEL_SIZE = 1.0 / (R - 1)
VDB_BAND_VOXELS = BAND / VOXEL_SIZE
LABEL_THRESHOLD = EPS - 1.0e-4

FINAL_EPS = 1.0 / FINAL_R
FINAL_VOXEL_SIZE = 1.0 / (FINAL_R - 1)


def configure_resolutions(
    input_res: int,
    final_res: int,
    proxy_points: int = 12_000_000,
    proxy_eps_scale: float = 1.0,
):
    """Configure the practical WTiVo quality/memory controls."""
    global R, FINAL_R, PROXY_POINTS, PROXY_EPS_SCALE
    global EPS, BAND, VOXEL_SIZE, VDB_BAND_VOXELS, LABEL_THRESHOLD
    global FINAL_EPS, FINAL_VOXEL_SIZE

    R = int(input_res)
    FINAL_R = int(final_res)
    PROXY_POINTS = int(proxy_points)
    PROXY_EPS_SCALE = float(proxy_eps_scale)

    if R < 2:
        raise ValueError("--input-res must be >= 2")
    if FINAL_R < 2:
        raise ValueError("--final-res must be >= 2")
    if PROXY_POINTS < 4:
        raise ValueError("--proxy-points must be >= 4")
    if PROXY_EPS_SCALE <= 0.0:
        raise ValueError("--proxy-eps-scale must be > 0")

    EPS = PROXY_EPS_SCALE / float(R)
    if EPS >= 0.5:
        raise ValueError("--proxy-eps-scale is too large for --input-res")
    if EPS <= 1.0e-4:
        raise ValueError(
            "--proxy-eps-scale/input-res must be greater than 1e-4 "
            "so the graph label threshold stays positive"
        )
    BAND = 3.0 / float(R)
    VOXEL_SIZE = 1.0 / float(R - 1)
    VDB_BAND_VOXELS = BAND / VOXEL_SIZE
    LABEL_THRESHOLD = EPS - 1.0e-4

    FINAL_EPS = 1.0 / float(FINAL_R)
    FINAL_VOXEL_SIZE = 1.0 / float(FINAL_R - 1)
# v6.3 FIX: tet-centroid label queries must use the legacy dense-era coordinate
# mapping directly. EPS is the geometry inset, NOT an additional label-query inset.
LABEL_QUERY_PADDING = 0.0
TOPO_UPLOAD_CHUNK = 262144  # 262k rows x 4 x int32 ~= 4 MiB host staging


def parse_args():
    p = argparse.ArgumentParser(
        description="WTiVo: WatertightVoxel Optimizer — sparse voxel proxy, tetra cell cut, CUDA graph optimization, and watertight FaithC-style finalization."
    )
    p.add_argument("--input")
    p.add_argument("--output")
    # Native-array bridge used by the ComfyUI MESH -> MESH node.  Raw .npy
    # files avoid a very large temporary GLB/OBJ conversion and keep the
    # standalone WTiVo Python/native environment isolated from ComfyUI's
    # Python/PyTorch ABI.
    p.add_argument("--input-vertices-npy", help=argparse.SUPPRESS)
    p.add_argument("--input-faces-npy", help=argparse.SUPPRESS)
    p.add_argument("--output-vertices-npy", help=argparse.SUPPRESS)
    p.add_argument("--output-faces-npy", help=argparse.SUPPRESS)
    p.add_argument(
        "--input-res", "--graph-res", dest="input_res", type=int, default=1536,
        help="WTiVo thick/input UDF + graph-label resolution. Default 1536. No artificial upper limit.",
    )
    p.add_argument(
        "--final-res", dest="final_res", type=int, default=1536,
        help="Final signed OpenVDB + FaithC reconstruction resolution. Default 1536. No artificial upper limit.",
    )
    p.add_argument(
        "--proxy_points", "--proxy-points", dest="proxy_points", type=int, default=12_000_000,
        help="Number of FaithC/QEF proxy points sent to CGAL. Default 12,000,000.",
    )
    p.add_argument(
        "--proxy_eps_scale", "--proxy-eps-scale", dest="proxy_eps_scale", type=float, default=1.0,
        help="Graph/proxy EPS scale relative to input resolution. Default 1.0.",
    )
    p.add_argument(
        "--proxy_feature_weight", type=float, default=1.5,
        help="Extra retention priority for local corners/creases/QEF disagreement. Default 1.5.",
    )
    p.add_argument("--lambda_fill", "--lamda_fill", dest="lambda_fill", type=float, default=10.0)
    p.add_argument(
        "--threads", type=int,
        default=int(os.environ.get("WTIVO_THREADS", os.cpu_count() or 1)),
        help="CPU threads. Default: all logical CPUs.",
    )
    p.add_argument(
        "--gpupr_local_steps", type=int, default=8,
        help="Legal local push/relabel steps per active CUDA vertex. Default 8.",
    )
    p.add_argument("--global_relabel_period", type=int, default=1024)
    p.add_argument("--max_rounds", type=int, default=2000000)
    p.add_argument(
        "--thick_band_voxels", type=float, default=3.0,
        help="Thick unsigned OpenVDB narrow-band width in voxels. Default 3.",
    )
    p.add_argument(
        "--thin_band_voxels", type=float, default=3.0,
        help="Final signed OpenVDB narrow-band width. Default 3 voxels.",
    )
    p.add_argument(
        "--thin_iso_vox", type=float, default=0.0,
        help="Final signed field iso in voxel units. Default 0.",
    )
    p.add_argument(
        "--faithc_component_mode", type=str, default="auto",
        choices=["auto", "keep_all", "largest"],
        help="Final component handling. auto preserves RAW when already watertight.",
    )
    p.add_argument(
        "--faithc_tri_mode", type=str, default="auto",
        choices=["auto", "simple_02", "simple_13", "length", "angle", "normal", "normal_abs"],
        help="FaithC quad triangulation rule. auto = normal_abs when normals are available.",
    )
    p.add_argument(
        "--faithc_clamp_anchors", type=int, choices=[0,1], default=1,
        help="Clamp QEF anchor to its source voxel. Default 1.",
    )
    p.add_argument("--faithc_lambda_n", type=float, default=1.0)
    p.add_argument(
        "--faithc_lambda_d", type=float, default=0.1,
        help="FaithC/QEF positional regularization. Default 0.1.",
    )
    # Compatibility only: v6.0 never calls OpenVDB VolumeToMesh, so this is ignored.
    p.add_argument("--vdb_adaptivity", type=float, default=0.0, help=argparse.SUPPRESS)
    return p.parse_args()


def cleanup_cuda(full=False):
    if full:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _gib(x):
    return float(x) / (1024.0 ** 3)


def compact_host_heap():
    """Best-effort Windows CRT heap compaction; never required for correctness."""
    if os.name != "nt":
        return
    for dll in ("ucrtbase", "msvcrt"):
        try:
            c = ctypes.CDLL(dll)
            fn = c._heapmin
            fn.restype = ctypes.c_int
            fn.argtypes = []
            fn()
            return
        except Exception:
            pass


def trim_working_set():
    """v6.30: deliberately disabled for speed.

    v6.28 repeatedly called Windows EmptyWorkingSet, forcing hot pages out of
    the process working set. Let Windows manage residency naturally instead.
    No mesh values or algorithmic decisions are changed.
    """
    return False


def mem_snapshot(tag: str):
    """Print accurate current-process private/RSS + system RAM + CUDA memory."""
    proc_ws = proc_private = sys_used = sys_free = commit_used = commit_free = 0
    try:
        if os.name == "nt":
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(st)
            if k32.GlobalMemoryStatusEx(ctypes.byref(st)):
                sys_free = st.ullAvailPhys
                sys_used = st.ullTotalPhys - st.ullAvailPhys
                commit_free = st.ullAvailPageFile
                commit_used = st.ullTotalPageFile - st.ullAvailPageFile

            pm = PROCESS_MEMORY_COUNTERS_EX()
            pm.cb = ctypes.sizeof(pm)
            h = k32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(h, ctypes.byref(pm), pm.cb):
                proc_ws = pm.WorkingSetSize
                proc_private = pm.PrivateUsage
    except Exception as e:
        print(f"[WTiVo-MEM] profiler warning: {e}", flush=True)

    gpu_text = "GPU n/a"
    try:
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            gpu_text = f"GPU used/free={_gib(total_b-free_b):.2f}/{_gib(free_b):.2f} GiB"
    except Exception:
        pass

    print(
        f"[WTiVo-MEM] {tag} | process private={_gib(proc_private):.2f} GiB | "
        f"working={_gib(proc_ws):.2f} GiB | system used/free={_gib(sys_used):.2f}/{_gib(sys_free):.2f} GiB | "
        f"commit used/free={_gib(commit_used):.2f}/{_gib(commit_free):.2f} GiB | {gpu_text}",
        flush=True,
    )



def topology_to_cuda_chunked(host_array, tag: str):
    """Upload an Nx4 int32 topology matrix to one contiguous CUDA tensor.

    Eigen/pybind arrays are often Fortran-contiguous.  A direct
    np.ascontiguousarray(host_array) would create another ~0.92 GiB host copy.
    v5.6 repacks only 262k rows at a time (~4 MiB host staging).
    """
    a = np.asarray(host_array)
    if a.ndim != 2 or a.shape[1] != 4:
        raise RuntimeError(f"{tag}: expected Nx4 topology array, got {a.shape}")
    n = int(a.shape[0])
    out = torch.empty((n, 4), dtype=torch.int32, device="cuda")
    t0 = time.perf_counter()
    for i in range(0, n, TOPO_UPLOAD_CHUNK):
        j = min(n, i + TOPO_UPLOAD_CHUNK)
        h = np.ascontiguousarray(a[i:j], dtype=np.int32)
        out[i:j].copy_(torch.from_numpy(h), non_blocking=False)
        del h
    torch.cuda.synchronize()
    print(
        f"[WTiVo-TOPOGPU] {tag} -> VRAM: rows={n:,} | "
        f"device={_gib(out.numel()*out.element_size()):.3f} GiB | "
        f"host staging <=~4 MiB | {time.perf_counter()-t0:.3f}s",
        flush=True,
    )
    return out


def load_mesh(
    path: Path = None, vertices_npy: Path = None, faces_npy: Path = None,
    vertices_array=None, faces_array=None,
):
    if vertices_array is not None or faces_array is not None:
        if vertices_array is None or faces_array is None:
            raise RuntimeError("Both direct vertex and face arrays are required")
        v = np.asarray(vertices_array)
        f = np.asarray(faces_array)
        if v.ndim != 2 or v.shape[1] != 3:
            raise RuntimeError(f"Direct vertices must be Nx3, got {v.shape}")
        if f.ndim != 2 or f.shape[1] != 3:
            raise RuntimeError(f"Direct faces must be Mx3 triangles, got {f.shape}")
        v = np.ascontiguousarray(v, dtype=np.float64)
        f = np.ascontiguousarray(f, dtype=np.int32)
        source_name = "ComfyUI native MESH arrays (in-process)"
    elif vertices_npy is not None or faces_npy is not None:
        if vertices_npy is None or faces_npy is None:
            raise RuntimeError("Both input native-array paths are required")
        v = np.load(str(vertices_npy), allow_pickle=False)
        f = np.load(str(faces_npy), allow_pickle=False)
        if v.ndim != 2 or v.shape[1] != 3:
            raise RuntimeError(f"Native vertices must be Nx3, got {v.shape}")
        if f.ndim != 2 or f.shape[1] != 3:
            raise RuntimeError(f"Native faces must be Mx3 triangles, got {f.shape}")
        v = np.ascontiguousarray(v, dtype=np.float64)
        f = np.ascontiguousarray(f, dtype=np.int32)
        source_name = "ComfyUI native MESH arrays"
    else:
        if path is None:
            raise RuntimeError("No input mesh was provided")
        mesh = trimesh.load_mesh(str(path), process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise RuntimeError(f"Could not load triangle mesh: {path}")
        v = np.asarray(mesh.vertices, dtype=np.float64)
        # Vertex count is far below INT32_MAX; keep face indices compact.
        f = np.asarray(mesh.faces, dtype=np.int32)
        source_name = str(path)
    if len(v) == 0 or len(f) == 0:
        raise RuntimeError(f"Empty mesh: {source_name}")
    if not np.isfinite(v).all():
        raise RuntimeError(f"Non-finite vertex coordinate in: {source_name}")
    if int(f.min()) < 0 or int(f.max()) >= len(v):
        raise RuntimeError(f"Face index outside vertex range in: {source_name}")
    return v, f


def _print_vdb_stats(tag: str, sparse):
    s = sparse.stats()
    print(
        f"[WTiVo-OpenVDB] {tag} | build={float(s['build_seconds']):.3f}s | "
        f"active_voxels={int(s['active_voxels']):,} | leaves={int(s['leaf_count']):,} | "
        f"grid_mem={int(s['memory_bytes'])/1024**3:.3f} GiB | "
        f"background={float(s['background']):.8g}",
        flush=True,
    )


def _build_sparse_udf(
    vertices: np.ndarray, faces: np.ndarray, bbox_min, bbox_max, threads: int, tag: str,
    signed_level_set: bool = False, band_voxels: float = VDB_BAND_VOXELS,
    geometry_eps: float = None, voxel_size: float = None,
):
    span = np.asarray(bbox_max, dtype=np.float64) - np.asarray(bbox_min, dtype=np.float64)
    if np.any(span <= 1e-20):
        raise RuntimeError(f"{tag}: degenerate bounding box")
    geps = float(EPS if geometry_eps is None else geometry_eps)
    vsize = float(VOXEL_SIZE if voxel_size is None else voxel_size)
    scale = 1.0 - 2.0 * geps
    t0 = time.perf_counter()
    vn = ((np.asarray(vertices, dtype=np.float64) - bbox_min) / span) * scale + geps
    vn = np.ascontiguousarray(vn, dtype=np.float32)
    fi = np.ascontiguousarray(faces, dtype=np.int32)
    print(
        f"[WTiVo-OpenVDB] {tag} indexed input | points={len(vn):,} faces={len(fi):,} | "
        f"host arrays={(vn.nbytes+fi.nbytes)/1024**3:.3f} GiB",
        flush=True,
    )
    # One VDB extension provides both the thick point-budget and thin finalization paths.
    # The single WTiVo VDB module provides both the fixed point-budget THICK path
    # and the manifold/watertight FaithC-style THIN finalizer.
    _backend_name = "WTiVo sparse OpenVDB/FaithC"
    print(f"[WTiVo-VDB] {tag} -> {_backend_name}", flush=True)
    sparse = vdb.SparseUDF(
        vn, fi, float(vsize), float(band_voxels),
        int(threads), bool(signed_level_set)
    )
    del vn, fi
    gc.collect()
    compact_host_heap()
    trim_working_set()
    _print_vdb_stats(tag, sparse)
    print(
        f"[WTiVo-OpenVDB] {tag} field={'SIGNED_LEVEL_SET' if signed_level_set else 'UNSIGNED_DISTANCE'} | "
        f"voxel={vsize:.9g} | geometry_eps={geps:.9g} | band_voxels={float(band_voxels):.4g}",
        flush=True,
    )
    print(f"[WTiVo-OpenVDB] {tag} total sparse-UDF build wrapper={time.perf_counter()-t0:.3f}s", flush=True)
    return sparse


def _faithc_extract_sparse(sparse, isovalue: float, threads: int, tri_mode: str, clamp_anchors: bool, lambda_n: float, lambda_d: float, tag: str):
    """Direct scalar-field -> FaithC-style contour tokens/QEF -> mesh. NEVER calls VolumeToMesh."""
    t0 = time.perf_counter()
    out_v, out_f, stats = sparse.faithc_mesh(
        float(isovalue), str(tri_mode), bool(clamp_anchors),
        float(lambda_n), float(lambda_d), int(threads),
    )
    out_v = np.ascontiguousarray(out_v, dtype=np.float32)
    out_f = np.ascontiguousarray(out_f, dtype=np.int32)
    print(
        f"[WTiVo-FaithC] {tag} | encode={float(stats['encode_seconds']):.3f}s | "
        f"decode={float(stats['decode_seconds']):.3f}s | active_cells={int(stats['active_cells']):,} | "
        f"quads={int(stats['quads']):,} | output v/f={len(out_v):,}/{len(out_f):,} | "
        f"VolumeToMesh_calls={int(stats['volume_to_mesh_calls'])}", flush=True,
    )
    if "manifold_split_cells" in stats:
        print(
            f"[FaithC-MANIFOLD-v6.21] split_cells={int(stats['manifold_split_cells']):,} | "
            f"extra_topology_vertices={int(stats['manifold_extra_vertices']):,} | "
            f"topology_vertices={int(stats['topology_vertices']):,}",
            flush=True,
        )
    if "repair_bad_before" in stats:
        print(
            f"[FaithC-FINALIZE-v6.21] bad_edges "
            f"{int(stats['repair_bad_before']):,}->{int(stats['repair_bad_after_split']):,}"
            f"->{int(stats['repair_bad_final']):,} | "
            f"before degree1={int(stats['repair_degree1_before']):,}, "
            f"degree>2={int(stats['repair_degree_gt2_before']):,}, "
            f"max_degree={int(stats['repair_max_degree_before'])} | "
            f"fan_duplicates={int(stats['repair_fan_duplicates']):,} | "
            f"tiny_loops_capped={int(stats['repair_loops_capped']):,} | "
            f"final degree1={int(stats['repair_degree1_final']):,}, "
            f"degree>2={int(stats['repair_degree_gt2_final']):,}",
            flush=True,
        )
    if len(out_v) == 0 or len(out_f) == 0:
        raise RuntimeError(f"{tag}: direct OpenVDB->FaithC bridge produced an empty mesh")
    print(f"[WTiVo-FaithC] {tag} total={time.perf_counter()-t0:.3f}s", flush=True)
    return out_v, out_f

def direct_thick_points(
    input_path: Path, proxy_feature_weight: float,
    threads: int, thick_band_voxels: float,
    clamp_anchors: bool, lambda_n: float, lambda_d: float,
    input_vertices_npy: Path = None, input_faces_npy: Path = None,
    input_vertices_array=None, input_faces_array=None,
):
    """Thick OpenVDB UDF -> direct FaithC/QEF anchors -> fixed point budget. NO proxy triangles."""
    t_total = time.perf_counter()
    v, f = load_mesh(input_path, input_vertices_npy, input_faces_npy, input_vertices_array, input_faces_array)
    print(f"[WTiVo-PointBudget] input v/f={len(v):,}/{len(f):,}", flush=True)
    bbox_min = v.min(0)
    bbox_max = v.max(0)

    sparse = _build_sparse_udf(
        v, f, bbox_min, bbox_max, threads, "thick",
        signed_level_set=False, band_voxels=thick_band_voxels,
    )
    del v, f
    gc.collect(); compact_host_heap(); trim_working_set()
    mem_snapshot("after thick sparse UDF build / point-budget path")

    t0 = time.perf_counter()
    out_v, stats = sparse.faithc_point_budget(
        float(EPS), int(PROXY_POINTS), bool(clamp_anchors),
        float(lambda_n), float(lambda_d),
        float(proxy_feature_weight), int(threads),
    )
    vertices = np.ascontiguousarray(out_v, dtype=np.float32)
    del out_v
    if len(vertices) == 0:
        raise RuntimeError("FaithC point-budget proxy produced zero points")

    print(
        f"[WTiVo-PointBudget] candidates={int(stats['candidate_points']):,} -> "
        f"selected={int(stats['selected_points']):,} | "
        f"encode={float(stats['encode_seconds']):.3f}s | "
        f"select={float(stats['select_seconds']):.3f}s | "
        f"feature_weight={float(stats['feature_weight']):g} | "
        f"strong_features={int(stats['strong_feature_candidates']):,}->"
        f"{int(stats['strong_feature_selected']):,} | "
        f"mean_feature={float(stats['mean_feature_candidates']):.4f}->"
        f"{float(stats['mean_feature_selected']):.4f} | "
        f"point_output={int(stats['output_bytes'])/1024**2:.1f} MiB",
        flush=True,
    )

    # Tested WTiVo bbox-fit behavior, but entirely on CPU.
    # No CUDA proxy mesh and no GPU decimator are created.
    bmin = vertices.min(axis=0).astype(np.float32)
    bmax = vertices.max(axis=0).astype(np.float32)
    denom = np.maximum(bmax - bmin, np.float32(1.0e-12))
    span = (np.asarray(bbox_max, dtype=np.float32) -
            np.asarray(bbox_min, dtype=np.float32))
    vertices -= bmin
    vertices /= denom
    vertices *= span
    vertices += np.asarray(bbox_min, dtype=np.float32)

    print(
        f"[WTiVo-PointBudget] CGAL proxy points={len(vertices):,} / target={PROXY_POINTS:,} | "
        f"proxy triangles=ZERO | GPU decimator=ZERO | "
        f"budget stage={time.perf_counter()-t0:.3f}s",
        flush=True,
    )
    print(
        f"[WTiVo-PointBudget] thick stage TOTAL={time.perf_counter()-t_total:.3f}s",
        flush=True,
    )
    mem_snapshot("after direct point-budget proxy / no CUDA decimation")
    return vertices, bbox_min, bbox_max, sparse


def native_watertight_count(faces: np.ndarray, threads: int, tag: str):
    """Exact native edge-degree check without Trimesh edge caches."""
    wt, bad_edges, wt_s, wt_bytes = core.is_watertight(
        np.ascontiguousarray(faces, dtype=np.int32), int(threads)
    )
    print(
        f"[{tag}] watertight={bool(wt)} | bad_edge_groups={int(bad_edges)} | "
        f"time={float(wt_s):.3f}s | edge_buffer={int(wt_bytes)/1024**2:.1f} MiB",
        flush=True,
    )
    return bool(wt), int(bad_edges)

def largest_component_same_rule(vertices: np.ndarray, faces: np.ndarray, threads: int):
    """Shared-edge component rule used by the tested production path."""
    t0 = time.perf_counter()
    ov, of, comps = core.largest_component(
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.int32),
        int(threads),
    )
    ov = np.asarray(ov, dtype=np.float32)
    of = np.asarray(of, dtype=np.int32)
    print(
        f"[WTiVo-Component] total={time.perf_counter()-t0:.3f}s | components={int(comps):,}",
        flush=True,
    )
    return ov, of

def direct_thin_mesh_owned(
    mesh_payload, threads: int, thin_iso_vox: float, thin_band_voxels: float,
    tri_mode: str, clamp_anchors: bool, lambda_n: float, lambda_d: float,
    component_mode: str,
):
    """Final SIGNED OpenVDB field -> direct FaithC-compatible contour. No VolumeToMesh."""
    t_total = time.perf_counter()
    vertices = np.asarray(mesh_payload.pop(0), dtype=np.float64)
    faces = np.asarray(mesh_payload.pop(0), dtype=np.int32)
    mesh_payload.clear()
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("thin bridge received an empty graph-cut mesh")

    v_min = vertices.min(0)
    v_max = vertices.max(0)
    sparse = _build_sparse_udf(
        vertices, faces, v_min, v_max, threads, "thin-final",
        signed_level_set=True, band_voxels=thin_band_voxels,
        geometry_eps=FINAL_EPS, voxel_size=FINAL_VOXEL_SIZE,
    )
    del vertices, faces
    gc.collect(); compact_host_heap(); trim_working_set()
    mem_snapshot("thin signed sparse field built / graph-cut arrays freed")

    thin_iso = float(thin_iso_vox) * float(FINAL_VOXEL_SIZE)
    print(
        f"[WTiVo-FaithC] thin SIGNED field | FINAL_R={FINAL_R} | iso_vox={thin_iso_vox:g} | "
        f"iso={thin_iso:.9g} | band_voxels={thin_band_voxels:g}", flush=True,
    )
    out_v, out_f = _faithc_extract_sparse(
        sparse, thin_iso, threads, tri_mode, clamp_anchors, lambda_n, lambda_d, "thin-final",
    )
    del sparse
    gc.collect(); compact_host_heap(); trim_working_set()
    mem_snapshot("after thin direct OpenVDB->FaithC / field destroyed")

    # Same v5.8 bbox restore so extractor is the isolated variable.
    omin = out_v.min(0)
    omax = out_v.max(0)
    denom = np.maximum(omax - omin, 1.0e-12)
    out_v = (out_v - omin) / denom
    out_v = out_v * (v_max - v_min) + v_min
    out_v = out_v.astype(np.float32, copy=False)

    # v6.19: do NOT assume the largest-component pass is topology-neutral.
    # Measure the raw manifold FaithC mesh first, then A/B against the filtered
    # mesh. AUTO keeps whichever has fewer bad edge groups.
    raw_wt, raw_bad = native_watertight_count(out_f, threads, "WTiVo-Audit")

    final_wt = raw_wt
    final_bad = raw_bad

    if component_mode == "keep_all":
        print("[WTiVo-Component] mode=keep_all -> preserving raw FaithC components", flush=True)

    elif component_mode == "auto" and raw_bad == 0:
        # Old AUTO rule only chooses filtered when filt_bad < raw_bad.
        # With raw_bad == 0, filtering cannot possibly win.
        print(
            "[WTiVo-Component] AUTO short-circuit: RAW bad_edges=0, "
            "so old AUTO must preserve RAW; skipping largest-component A/B",
            flush=True,
        )

    else:
        filtered_v, filtered_f = largest_component_same_rule(out_v, out_f, threads)
        filt_wt, filt_bad = native_watertight_count(
            filtered_f, threads, "WTiVo-Audit-PostComponent"
        )

        choose_filtered = (component_mode == "largest")
        if component_mode == "auto":
            if raw_bad >= 0 and filt_bad >= 0:
                choose_filtered = filt_bad < raw_bad
            elif filt_bad >= 0 and raw_bad < 0:
                choose_filtered = True
            else:
                choose_filtered = False

        if choose_filtered:
            print(
                f"[WTiVo-Component] selected=LARGEST | bad_edges {raw_bad}->{filt_bad}",
                flush=True,
            )
            out_v, out_f = filtered_v, filtered_f
            final_wt, final_bad = filt_wt, filt_bad
        else:
            print(
                f"[WTiVo-Component] selected=RAW-FAITHC | bad_edges raw={raw_bad}, largest={filt_bad}",
                flush=True,
            )
            del filtered_v, filtered_f

    print(f"[WTiVo-Final] thin direct-FaithC TOTAL: {time.perf_counter()-t_total:.3f}s", flush=True)
    return out_v, out_f, final_wt, final_bad

def gpupr_graph_cut_fast_v630(
    tet_verts, tets_cuda, neighbors_cuda, labels, fill_T: float, threads: int,
    global_relabel_period: int, max_rounds: int, local_steps: int,
):
    t0 = time.perf_counter()
    result = gpupr.graph_cut_fast(
        np.ascontiguousarray(np.asarray(tet_verts, dtype=np.float64)),
        tets_cuda,
        neighbors_cuda,
        np.ascontiguousarray(np.asarray(labels, dtype=np.uint8).reshape(-1)),
        float(fill_T),
        int(threads),
        int(global_relabel_period),
        int(max_rounds),
        int(local_steps),
    )

    (
        full_labels,
        adjusted_flow,
        raw_flow,
        graph_bytes,
        workspace_bytes,
        full_edges,
        free_edges,
        fixed_source,
        fixed_sink,
        terminal_edges,
        used_threads,
        macro_rounds,
        global_relabels,
        mapping_peak_bytes,
        build_upload_seconds,
        used_local_steps,
        solve_seconds,
        native_total_seconds,
    ) = result

    full_labels = np.asarray(full_labels, dtype=np.uint8)

    print(
        f"[WTiVo-GPUPr] device graph={int(graph_bytes)/1024**3:.3f} GiB | "
        f"solver workspace={int(workspace_bytes)/1024**3:.3f} GiB",
        flush=True,
    )
    print(
        f"[WTiVo-GPUPr] solve={float(solve_seconds):.3f}s | "
        f"macro_rounds={int(macro_rounds):,} | local_steps={int(used_local_steps)} | "
        f"global_relabels={int(global_relabels)}",
        flush=True,
    )
    print(
        f"[WTiVo-GPUPr] flow adjusted/raw={int(adjusted_flow)}/{int(raw_flow)} | "
        f"graph build={float(build_upload_seconds):.3f}s | "
        f"native total={float(native_total_seconds):.3f}s | "
        f"wrapper total={time.perf_counter()-t0:.3f}s",
        flush=True,
    )
    return np.ascontiguousarray(full_labels, dtype=np.uint8)

def main():
    a = parse_args()
    native_input = bool(a.input_vertices_npy or a.input_faces_npy)
    native_output = bool(a.output_vertices_npy or a.output_faces_npy)
    if native_input:
        if not (a.input_vertices_npy and a.input_faces_npy):
            raise SystemExit("Native-array input requires both --input-vertices-npy and --input-faces-npy")
        if a.input:
            raise SystemExit("Use either --input or native-array input, not both")
    elif not a.input:
        raise SystemExit("--input is required for file mode")
    if native_output:
        if not (a.output_vertices_npy and a.output_faces_npy):
            raise SystemExit("Native-array output requires both --output-vertices-npy and --output-faces-npy")
        if a.output:
            raise SystemExit("Use either --output or native-array output, not both")
    elif not a.output:
        raise SystemExit("--output is required for file mode")
    if not (1 <= int(a.gpupr_local_steps) <= 32):
        raise ValueError("--gpupr_local_steps must be between 1 and 32")
    configure_resolutions(a.input_res, a.final_res, a.proxy_points, a.proxy_eps_scale)
    if a.vdb_adaptivity != 0.0:
        print("[WARN] --vdb_adaptivity is ignored because VolumeToMesh is never called.", flush=True)
    if a.thin_band_voxels <= 1.8:
        raise SystemExit("--thin_band_voxels should be > 1.8 for complete direct contour cells; use 3 by default")
    if a.thick_band_voxels <= 2.75:
        raise SystemExit("--thick_band_voxels must be > 2.75 for iso~1 + full cube support; use ~3 by default")
    if a.faithc_lambda_n < 0 or a.faithc_lambda_d <= 0:
        raise SystemExit("FaithC lambda_n must be >=0 and lambda_d must be >0")
    if a.proxy_feature_weight < 0.0:
        raise ValueError("--proxy_feature_weight must be >= 0")
    if a.lambda_fill < 0:
        raise ValueError("--lambda_fill must be >=0")
    if a.threads < 1:
        raise ValueError("--threads must be >=1")
    if os.name != "nt":
        raise SystemExit("WTiVo 1.1 currently supports Windows 10/11 x64 only.")
    if not torch.cuda.is_available():
        raise SystemExit("WTiVo requires an NVIDIA CUDA-capable GPU. Run Setup-Windows.cmd first.")

    inp = None
    input_vertices_npy = input_faces_npy = None
    if native_input:
        input_vertices_npy = Path(a.input_vertices_npy).expanduser().resolve()
        input_faces_npy = Path(a.input_faces_npy).expanduser().resolve()
        for array_path in (input_vertices_npy, input_faces_npy):
            if not array_path.exists():
                raise FileNotFoundError(array_path)
        input_display = "ComfyUI native MESH arrays"
    else:
        inp = Path(a.input).expanduser().resolve()
        if not inp.exists():
            raise FileNotFoundError(inp)
        input_display = str(inp)

    out = None
    output_vertices_npy = output_faces_npy = None
    if native_output:
        output_vertices_npy = Path(a.output_vertices_npy).expanduser().resolve()
        output_faces_npy = Path(a.output_faces_npy).expanduser().resolve()
        output_vertices_npy.parent.mkdir(parents=True, exist_ok=True)
        output_faces_npy.parent.mkdir(parents=True, exist_ok=True)
        output_display = "ComfyUI native MESH arrays"
    else:
        out = Path(a.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        output_display = str(out)

    print("=======================================================")
    print("  WTiVo 1.1 — WatertightVoxel Optimizer")
    print("=======================================================")
    print(f"Input          : {input_display}")
    print(f"Output         : {output_display}")
    print(f"Input/graph R  : {R} sparse global OpenVDB UDF")
    print(f"Final FaithC R : {FINAL_R} signed sparse OpenVDB")
    print(f"Proxy budget   : {PROXY_POINTS:,} points [FIXED]")
    print(f"Feature weight : {a.proxy_feature_weight:g}")
    print(f"Lambda fill    : {a.lambda_fill}")
    print(f"CPU threads    : {a.threads}")
    print("[NO TILE] one global sparse OpenVDB domain; no slabs/chunks in the volume")
    print("[THICK] unsigned OpenVDB field -> DIRECT FaithC tokens/QEF at iso=1/R; NO VolumeToMesh")
    print(f"[THIN] signed OpenVDB field -> DIRECT FaithC tokens/QEF | iso_vox={a.thin_iso_vox:g} | band_voxels={a.thin_band_voxels:g}")
    print(
        f"[FIELD] graph_voxel={VOXEL_SIZE:.8g}; graph_eps={EPS:.8g}; "
        f"final_voxel={FINAL_VOXEL_SIZE:.8g}; final_eps={FINAL_EPS:.8g}; "
        f"thick_band_vox={a.thick_band_voxels:g}; thin_iso_vox={a.thin_iso_vox:g}; "
        f"thin_band_vox={a.thin_band_voxels:g}"
    )
    print(f"[LABEL-FIX] tet centroid query_padding={LABEL_QUERY_PADDING:g} (legacy dense-era coordinates; EPS is NOT applied twice)")
    print(f"[FAITHC] tri_mode={a.faithc_tri_mode} | clamp={a.faithc_clamp_anchors} | lambda_n={a.faithc_lambda_n:g} | lambda_d={a.faithc_lambda_d:g}")
    print("[EXTRACTOR] VolumeToMesh calls: ZERO (thick=FaithC bridge, thin=FaithC bridge)")
    print("[MEM] sparse OpenVDB fields + CPU streaming FaithC-compatible decoder + GPU graph topology")
    print("[GRAPH] Exact ReducedGraph + CUDA Push-Relabel Phase-I; periodic global relabel enabled")
    print("[CELL-CUT] CGAL Delaunay tetrahedralization + tetra-neighbor graph + CelloCut-compatible lambda_fill objective")
    print("[POINT-PROXY] FaithC/QEF anchors -> exact point budget; NO proxy triangles; NO GPU decimator")
    print("[WTiVo] THICK+LABELS = sparse OpenVDB + direct fixed point budget")
    print("[WTiVo] TETRA = CGAL Parallel_tag + direct neighbor export")
    print(f"[WTiVo] GRAPH = exact CUDA multi-discharge Push-Relabel | local_steps={a.gpupr_local_steps}")
    print("[WTiVo] THIN = signed sparse OpenVDB + manifold FaithC-style finalizer")
    print("[WTiVo] thick UDF freed immediately after labels")
    print("[WTiVo] Windows EmptyWorkingSet disabled; CRT heap compaction retained")
    print("[WTiVo] final watertight validation always ON exactly once")
    print(f"[PROXY-BUDGET] {PROXY_POINTS:,} points are sent to CGAL")
    print("[NO-RATIO] --decimate_ratio remains intentionally unavailable")
    print(f"[LABEL-FIX] geometry EPS={PROXY_EPS_SCALE:g}/R; tet centroid OpenVDB query padding remains 0")
    print(f"[RES-CONTROL] --input-res={R} controls thick field + cell-label geometry")
    print(f"[RES-CONTROL] --final-res={FINAL_R} controls ONLY final signed OpenVDB + FaithC density")
    print(f"[PROXY-BUDGET] CGAL target={PROXY_POINTS:,} points regardless of resolution")
    print("[NO FINAL DECIMATOR] polygon count is the direct FaithC contour at --final-res")
    print("[NO RES CAP] --input-res and --final-res have NO artificial upper limit")
    print("[WARNING] practical limit is only RAM/VRAM/runtime; very high final-res can create enormous meshes")
    print()

    print(f"[TORCH] {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[CORE] {getattr(core, '__file__', '<unknown>')}")
    print(f"[VDB] {getattr(vdb, '__file__', '<unknown>')}")
    print(f"[GPUPr] {getattr(gpupr, '__file__', '<unknown>')}")

    t_all = time.perf_counter()
    mem_snapshot("START")

    # 1. Thick geometry from a global sparse OpenVDB unsigned-distance field.
    thick_v, origin_min, origin_max, sparse_udf = direct_thick_points(
        inp, a.proxy_feature_weight,
        a.threads, a.thick_band_voxels,
        bool(a.faithc_clamp_anchors), a.faithc_lambda_n, a.faithc_lambda_d,
        input_vertices_npy, input_faces_npy,
    )
    mem_snapshot("after direct point proxy / sparse UDF retained for labels")
    gc.collect()
    cleanup_cuda(full=False)
    trim_working_set()
    mem_snapshot("before tetra / point-only proxy + sparse UDF retained on CPU")

    # 2. Parallel tetrahedralization. CGAL exports the same exact topology.
    print("Tetrahedralizing + exporting neighbors...", flush=True)
    t0 = time.perf_counter()
    thick_v64 = np.ascontiguousarray(thick_v, dtype=np.float64)
    del thick_v
    gc.collect(); compact_host_heap(); trim_working_set()
    print(
        f"[WTiVo-PointBudget] CGAL input float64={len(thick_v64):,} points | "
        f"{thick_v64.nbytes/1024**3:.3f} GiB | faces=ZERO",
        flush=True,
    )
    tet_verts, tets, neighbors = core.tetrahedralize_neighbors(thick_v64, int(a.threads))
    print(f"[WTiVo-Timing] tetrahedralize+neighbors: {time.perf_counter()-t0:.3f}s", flush=True)
    del thick_v64
    gc.collect()
    cleanup_cuda(full=True)
    compact_host_heap()
    mem_snapshot("after tetra+neighbors / both topology tables still host")

    # 3. Sparse label sampling. Keep only TETS on host for this step; move
    # NEIGHBORS to VRAM immediately.
    #
    # v6.3 correctness fix:
    #   The old dense label path sampled tetrahedron centroids directly in bbox
    #   coordinates. Earlier OpenVDB runners incorrectly applied EPS again to the
    #   query coordinates. The v6.2 A/B test showed that this changed ~11.86% of
    #   initial labels and created the RAW_GRAPH surface noise.
    #   Keep geometry inset EPS, but label-query padding is exactly ZERO.
    neighbors_cuda = topology_to_cuda_chunked(neighbors, "neighbors")
    del neighbors
    gc.collect()
    compact_host_heap()
    trim_working_set()
    mem_snapshot("host NEIGHBORS freed; sparse UDF + host tets remain for labels")

    t0 = time.perf_counter()
    initial_labels, vdb_label_seconds = sparse_udf.sample_tet_labels(
        np.ascontiguousarray(tet_verts, dtype=np.float64),
        np.ascontiguousarray(tets, dtype=np.int32),
        np.ascontiguousarray(np.asarray(origin_min, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(origin_max, dtype=np.float64)),
        float(LABEL_QUERY_PADDING), float(LABEL_THRESHOLD), int(a.threads),
    )
    initial_labels = np.ascontiguousarray(np.asarray(initial_labels, dtype=np.uint8).reshape(-1))
    print(
        f"[WTiVo-OpenVDB] tet centroid BoxSampler labels={len(initial_labels):,} | "
        f"query_padding={LABEL_QUERY_PADDING:g} | threshold={LABEL_THRESHOLD:.8g} | "
        f"total={float(vdb_label_seconds):.3f}s",
        flush=True,
    )
    del sparse_udf
    gc.collect()
    compact_host_heap()
    trim_working_set()
    mem_snapshot("after sparse labels / OpenVDB grid freed / host tets still alive")

    tets_cuda = topology_to_cuda_chunked(tets, "tets")
    del tets
    gc.collect()
    compact_host_heap()
    trim_working_set()
    mem_snapshot("ALL full host topology freed / tets+neighbors GPU-resident")

    # 4. Both full topology tables are already GPU-only.
    print("[WTiVo-Graph] full topology streamed from VRAM in <=~8 MiB host chunks", flush=True)

    # 5. Exact CelloCut-compatible graph capacities + WTiVo CUDA multi-discharge scheduling.
    print("Graph cutting (WTiVo exact reduced graph + CUDA multi-discharge)...", flush=True)
    t0 = time.perf_counter()
    mem_snapshot("before GPU graph cut / no full host topology")
    new_labels = gpupr_graph_cut_fast_v630(
        tet_verts, tets_cuda, neighbors_cuda, initial_labels,
        a.lambda_fill, a.threads, a.global_relabel_period,
        a.max_rounds, a.gpupr_local_steps,
    )
    mem_snapshot("after GPU graph cut / reduced graph freed / full topology still GPU")
    print(f"[WTiVo-Timing] graph_cut wrapper: {time.perf_counter()-t0:.3f}s", flush=True)
    del initial_labels
    gc.collect()

    # 6. Extract the exact cut surface while the full topology remains in VRAM.
    # The native extractor streams only small topology chunks to CPU.  It emits
    # each cut face from its INSIDE tet, which is the same boundary set and uses
    # the same inside-tet orientation rule as ThreadPack v3.
    t0 = time.perf_counter()
    gc_v, gc_f = gpupr.surface_extraction_topology_cuda(
        new_labels, tet_verts, tets_cuda, neighbors_cuda, int(a.threads)
    )
    print(f"[WTiVo-Timing] surface extraction: {time.perf_counter()-t0:.3f}s", flush=True)
    del new_labels, tet_verts, tets_cuda, neighbors_cuda
    gc.collect()
    cleanup_cuda(full=True)
    compact_host_heap()
    trim_working_set()
    mem_snapshot("after GPU topology freed / graph stage fully released")

    # 7. Final SIGNED OpenVDB field, then direct FaithC-compatible tokens/QEF. No VolumeToMesh.
    thin_payload = [gc_v, gc_f]
    del gc_v, gc_f
    final_v, final_f, final_wt_known, final_bad_known = direct_thin_mesh_owned(
        thin_payload, a.threads, a.thin_iso_vox, a.thin_band_voxels,
        a.faithc_tri_mode, bool(a.faithc_clamp_anchors),
        a.faithc_lambda_n, a.faithc_lambda_d,
        a.faithc_component_mode,
    )
    gc.collect()
    compact_host_heap()

    # The exact checker already ran once on the final face array inside
    # direct_thin_mesh_owned, after the v6.21 FaithC finalizer.
    watertight = bool(final_wt_known)
    bad_edges = int(final_bad_known)

    print("-------------------------------------------------------")
    print("  WTiVo FINAL WATERTIGHT RESULT")
    print("-------------------------------------------------------")
    print(f"[FINAL] v/f={len(final_v):,}/{len(final_f):,}")
    print(f"[FINAL] watertight={watertight} | bad_edge_groups={bad_edges}")
    mem_snapshot("FINAL arrays only / before output")

    if native_output:
        np.save(
            str(output_vertices_npy),
            np.ascontiguousarray(final_v, dtype=np.float32),
            allow_pickle=False,
        )
        np.save(
            str(output_faces_npy),
            np.ascontiguousarray(final_f, dtype=np.int32),
            allow_pickle=False,
        )
    else:
        mesh = trimesh.Trimesh(vertices=final_v, faces=final_f, process=False)
        mesh.export(str(out))
        del mesh
    gc.collect()
    compact_host_heap()
    mem_snapshot("after export")

    print(f"[DONE] total={(time.perf_counter()-t_all)/60.0:.2f} min")
    print(f"[DONE] {output_display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
