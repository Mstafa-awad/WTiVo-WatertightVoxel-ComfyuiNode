from __future__ import annotations
import logging
import os
import subprocess
import sys
import tempfile
import time
import numpy as np

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
    """Run WTiVo in an isolated subprocess and return output arrays + audit info."""
    if faithc_component_mode not in ("auto", "keep_all", "largest"):
        raise ValueError("faithc_component_mode must be auto, keep_all, or largest")
    if float(proxy_feature_weight) < 0.0:
        raise ValueError("proxy_feature_weight must be >= 0")
    if int(threads) < 1:
        raise ValueError("threads must be >= 1")

    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"WTiVo vertices must be Nx3, got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"WTiVo faces must be Mx3, got {faces.shape}")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("WTiVo received an empty mesh")

    # Create a secure temporary directory for the NPY bridge
    with tempfile.TemporaryDirectory(prefix="wtivo_") as tmp_dir:
        v_in_path = os.path.join(tmp_dir, "v_in.npy")
        f_in_path = os.path.join(tmp_dir, "f_in.npy")
        v_out_path = os.path.join(tmp_dir, "v_out.npy")
        f_out_path = os.path.join(tmp_dir, "f_out.npy")
        
        # Save input arrays for the subprocess to read
        np.save(v_in_path, np.ascontiguousarray(vertices, dtype=np.float64))
        np.save(f_in_path, np.ascontiguousarray(faces, dtype=np.int32))
        
        # Locate the standalone wtivo.py script
        wtivo_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wtivo.py")
        if not os.path.exists(wtivo_script):
            raise RuntimeError(f"Could not find wtivo.py at {wtivo_script}")

        # Build the command line arguments matching wtivo.py's argparse
        cmd = [
            sys.executable, wtivo_script,
            "--input-vertices-npy", v_in_path,
            "--input-faces-npy", f_in_path,
            "--output-vertices-npy", v_out_path,
            "--output-faces-npy", f_out_path,
            "--input-res", str(int(input_res)),
            "--final-res", str(int(final_res)),
            "--proxy_points", str(int(proxy_points)),
            "--proxy_eps_scale", str(float(proxy_eps_scale)),
            "--proxy_feature_weight", str(float(proxy_feature_weight)),
            "--lambda_fill", str(float(lambda_fill)),
            "--threads", str(int(threads)),
            "--thick_band_voxels", str(THICK_BAND_VOXELS),
            "--thin_band_voxels", str(THIN_BAND_VOXELS),
            "--thin_iso_vox", str(float(thin_iso_vox)),
            "--faithc_component_mode", str(faithc_component_mode),
            "--faithc_tri_mode", FAITHC_TRI_MODE,
            "--faithc_clamp_anchors", str(int(FAITHC_CLAMP_ANCHORS)),
            "--faithc_lambda_n", str(FAITHC_LAMBDA_N),
            "--faithc_lambda_d", str(FAITHC_LAMBDA_D),
        ]
        
        logging.info("[WTiVo] Spawning isolated subprocess to prevent native memory leaks...")
        t_all = time.perf_counter()
        
        # Run the subprocess, streaming output to the ComfyUI console in real-time
        process = subprocess.Popen(
            cmd,
            env={**os.environ, "WTIVO_SUBPROCESS": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )
        
        output_lines = []
        try:
            for line in process.stdout:
                print(line, end="", flush=True) # Print to ComfyUI console
                output_lines.append(line)
        except Exception as e:
            logging.warning(f"[WTiVo] Warning while reading subprocess stdout: {e}")
            
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(
                f"WTiVo subprocess crashed with exit code {process.returncode}. "
                "Check the console above for C++ or CUDA errors."
            )
            
        # Parse the final stats from the captured stdout
        watertight = False
        bad_edges = 0
        
        for line in output_lines:
            if "[FINAL] watertight=" in line:
                try:
                    # Example: [FINAL] watertight=True | bad_edge_groups=0
                    parts = line.split("|")
                    wt_str = parts[0].split("=")[1].strip()
                    watertight = (wt_str == "True")
                    be_str = parts[1].split("=")[1].strip()
                    bad_edges = int(be_str)
                except Exception:
                    pass
                    
        total_time = time.perf_counter() - t_all
        
        # Load the results
        if not os.path.exists(v_out_path) or not os.path.exists(f_out_path):
            raise RuntimeError("WTiVo subprocess finished but failed to write output NPY files.")
            
        final_v = np.load(v_out_path)
        final_f = np.load(f_out_path)
        
        logging.info(
            "[WTiVo] Done: %s vertices / %s faces | watertight=%s | %.2fs (Subprocess)",
            f"{len(final_v):,}",
            f"{len(final_f):,}",
            bool(watertight),
            total_time,
        )
        
        return final_v, final_f, bool(watertight), int(bad_edges), float(total_time)