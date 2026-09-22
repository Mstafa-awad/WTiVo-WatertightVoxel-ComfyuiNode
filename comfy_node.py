# SPDX-License-Identifier: GPL-3.0-or-later
"""WTiVo ComfyUI native MESH -> MESH node without RAM cleaning."""

from __future__ import annotations

import os

import numpy as np
import torch

from .inprocess import process_arrays


DEFAULT_THREADS = max(1, os.cpu_count() or 1)
INT_MAX = 2_147_483_647


def _mesh_to_numpy(mesh):
    vertices = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)

    if not torch.is_tensor(vertices) or not torch.is_tensor(faces):
        raise TypeError("WTiVo requires the official ComfyUI MESH object.")

    if vertices.ndim == 2:
        vertices = vertices.unsqueeze(0)
    if faces.ndim == 2:
        faces = faces.unsqueeze(0)

    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"Expected MESH vertices [B,N,3], got {tuple(vertices.shape)}")
    if faces.ndim != 3 or faces.shape[-1] != 3:
        raise ValueError(f"Expected triangular MESH faces [B,M,3], got {tuple(faces.shape)}")
    if vertices.shape[0] != 1 or faces.shape[0] != 1:
        raise ValueError("WTiVo processes one mesh at a time. Use batch size 1.")

    vertex_count = int(vertices.shape[1])
    face_count = int(faces.shape[1])

    vertex_counts = getattr(mesh, "vertex_counts", None)
    face_counts = getattr(mesh, "face_counts", None)

    if vertex_counts is not None:
        vertex_count = int(vertex_counts[0].item())
    if face_counts is not None:
        face_count = int(face_counts[0].item())

    v = (
        vertices[0, :vertex_count]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
    )
    f = (
        faces[0, :face_count]
        .detach()
        .to(device="cpu", dtype=torch.int32)
        .contiguous()
        .numpy()
    )

    if len(v) == 0 or len(f) == 0:
        raise ValueError("WTiVo received an empty mesh.")
    if not np.isfinite(v).all():
        raise ValueError("WTiVo received NaN or infinite vertex coordinates.")
    if int(f.min()) < 0 or int(f.max()) >= len(v):
        raise ValueError("WTiVo received invalid face indices.")

    return v, f


def _mesh_counts(mesh):
    vertex_count = int(mesh.vertices.shape[-2])
    face_count = int(mesh.faces.shape[-2])

    vertex_counts = getattr(mesh, "vertex_counts", None)
    face_counts = getattr(mesh, "face_counts", None)

    if vertex_counts is not None:
        vertex_count = int(vertex_counts[0].item())
    if face_counts is not None:
        face_count = int(face_counts[0].item())

    return vertex_count, face_count


class WTiVoNativeMeshToMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "input_res": (
                    "INT",
                    {
                        "default": 1536,
                        "min": 256,
                        "max": 8192,
                        "step": 128,
                    },
                ),
                "final_res": (
                    "INT",
                    {
                        "default": 1536,
                        "min": 256,
                        "max": 8192,
                        "step": 128,
                    },
                ),
                "proxy_points": (
                    "INT",
                    {
                        "default": 12_000_000,
                        "min": 0,
                        "max": 100_000_000,
                        "step": 1_000_000,
                    },
                ),
                "proxy_eps_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.25,
                        "max": 4.0,
                        "step": 0.05,
                    },
                ),
                "proxy_feature_weight": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.05,
                    },
                ),
                "lambda_fill": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 1.0,
                    },
                ),
                "threads": (
                    "INT",
                    {
                        "default": DEFAULT_THREADS,
                        "min": 1,
                        "max": INT_MAX,
                        "step": 1,
                    },
                ),
                "thin_iso_vox": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                    },
                ),
                "faithc_component_mode": (
                    ["auto", "keep_all", "largest"],
                    {"default": "keep_all"},
                ),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "execute"
    CATEGORY = "3d/mesh/WTiVo"
    DESCRIPTION = (
        "WTiVo watertight reconstruction for native ComfyUI/Trellis MESH. "
        "Runs in-process without RAM cleanup."
    )

    def execute(
        self,
        mesh,
        input_res: int,
        final_res: int,
        proxy_points: int,
        proxy_eps_scale: float,
        proxy_feature_weight: float,
        lambda_fill: float,
        threads: int,
        thin_iso_vox: float,
        faithc_component_mode: str,
    ):
        mesh_class = type(mesh)

        vertex_count, face_count = _mesh_counts(mesh)
        print(
            f"[WTiVo] Input: {vertex_count:,} vertices / {face_count:,} faces | "
            f"R={int(input_res)} -> {int(final_res)} | proxy={int(proxy_points):,}"
        )

        final_v, final_f, _watertight, _bad_edges, _total = process_arrays(
            *_mesh_to_numpy(mesh),
            input_res=int(input_res),
            final_res=int(final_res),
            proxy_points=int(proxy_points),
            proxy_eps_scale=float(proxy_eps_scale),
            proxy_feature_weight=float(proxy_feature_weight),
            lambda_fill=float(lambda_fill),
            threads=int(threads),
            thin_iso_vox=float(thin_iso_vox),
            faithc_component_mode=str(faithc_component_mode),
        )

        vertices_out = torch.as_tensor(final_v, dtype=torch.float32)
        faces_out = torch.as_tensor(final_f, dtype=torch.int32)

        try:
            output_mesh = mesh_class(
                vertices=vertices_out.unsqueeze(0),
                faces=faces_out.unsqueeze(0),
            )
        except TypeError as exc:
            raise TypeError(
                "WTiVo could not construct the official ComfyUI MESH output. "
                "Update ComfyUI to the current version."
            ) from exc

        return (output_mesh,)


NODE_CLASS_MAPPINGS = {
    "WTiVoNativeMeshToMesh": WTiVoNativeMeshToMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WTiVoNativeMeshToMesh": "WTiVo - Mesh Watertight",
}