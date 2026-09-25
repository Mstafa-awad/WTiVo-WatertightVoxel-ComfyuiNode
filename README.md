# WTiVo — WatertightVoxel (ComfyUI Node)

A high-performance, in-process watertight remeshing node for **ComfyUI**. WTiVo converts defective, non-manifold, or open 3D triangle meshes (such as raw outputs from AI 3D generators like TRELLIS) into dense, closed, manifold meshes using sparse voxel fields, tetrahedral cell cuts, CUDA graph optimization, and manifold contouring.

This edition features a **prebuilt in-process backend**, eliminating local C++/CUDA compilation entirely.

---

## ⚡ Key Highlights & Changes

* **Native ComfyUI Custom Node:** Converted from a standalone CLI script into a native, in-process ComfyUI node (`WTiVo - Mesh Watertight`), executing directly on native `MESH` objects.
* **Prebuilt Backend (No Compilation Required):** Ships with precompiled CPython 3.12 / CUDA 12.8 `.pyd` native extensions. No MSVC, CMake, CGAL, or manual CUDA compilation steps (`Setup-Windows.cmd`) are needed.
* **In-Memory Zero-IPC Pipeline:** Automatically unloads active upstream models before processing and passes native PyTorch tensors and NumPy arrays directly in memory—no temporary GLB/OBJ files or subprocess bridges.
* **CUDA-Accelerated Graph Cut:** Leverages multi-discharge Push-Relabel CUDA graph optimization and CGAL Delaunay tetrahedralization for fast watertight surface extraction.
* **Exact Edge Validation:** Guarantees topological integrity with built-in audits verifying exact degree-2 edge groups (`watertight=True`).

---

## 📋 System Requirements

* **OS:** Windows 10 / 11 x64
* **GPU:** NVIDIA CUDA-capable GPU (RTX 20 / 30 / 40 / 50 Series; ~8 GB+ VRAM recommended)
* **Environment:** ComfyUI with **Python 3.12 embedded**
* **PyTorch Requirement:** Requires **PyTorch 2.8.0 with CUDA 12.8** installed in your embedded Python environment:
  ```text
  torch-2.8.0+cu128-cp312-cp312-win_amd64.whl
  ```
  *(Located in `\ComfyUI-Easy-Install\python_embeded\` or your custom embedded environment)*

---

## 📦 Installation

1. **Verify PyTorch Version:** Ensure your embedded Python environment has `torch 2.8.0+cu128` installed. For ComfyUI Easy Install environments, install the wheel directly into `python_embeded`:
   ```cmd
   .\python_embeded\python.exe -m pip install torch-2.8.0+cu128-cp312-cp312-win_amd64.whl
   ```

2. **Install Custom Node:** Clone or extract this repository into your ComfyUI custom nodes directory:
   ```cmd
   cd ComfyUI\custom_nodes\
   git clone https://github.com/Mstafa-awad/WTiVo-WatertightVoxel-ComfyuiNode.git
   ```

3. **Restart ComfyUI:** Launch ComfyUI. The node will be available under `3d/mesh/WTiVo` as **WTiVo - Mesh Watertight**.

---

## ⚙️ Node Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :---: | :--- |
| `mesh` | **MESH** | — | Input ComfyUI native mesh object (e.g., from `VaeDecodeShapeTrellis`). |
| `input_res` | **INT** | `1536` | Resolution for the initial thick UDF sparse field and graph-labeling pass. |
| `final_res` | **INT** | `1536` | Resolution for the final signed OpenVDB field and FaithC reconstruction. |
| `proxy_points` | **INT** | `12000000` | QEF proxy point budget sent to CGAL for tetrahedralization. |
| `proxy_eps_scale` | **FLOAT** | `1.0` | Inset scale for graph proxy geometry relative to resolution. |
| `proxy_feature_weight` | **FLOAT** | `1.5` | Priority weight for preserving sharp corners, edges, and creases. |
| `lambda_fill` | **FLOAT** | `20.0` | Regularization factor for tetrahedral cell-cut fill. |
| `thin_iso_vox` | **FLOAT** | `0.0` | Surface offset in voxel units for the final signed field. |
| `faithc_component_mode` | **ENUM** | `largest` | Component handling (`auto`, `keep_all`, `largest`). `largest` is the best option to make it watertight. |
| `threads` | **INT** | *Max CPU* | Number of logical CPU threads allocated for parallel operations. |

---

## 🔄 Workflow Integration

```text
[ VaeDecodeShapeTrellis ]
          │ (MESH)
          ▼
[ WTiVo - Mesh Watertight ]
          │ (Watertight MESH)
          ▼
[ Preview 3D / Save 3D / Exporter ]
```

*Note: WTiVo completely rebuilds topology to ensure strict watertight geometry. Input UVs, textures, and material maps are not preserved. Ensure batch size is set to 1.*

---

## ⚖️ License & Attribution

* **License:** GNU General Public License v3.0 or later (**GPL-3.0-or-later**).
* **Attribution:** WTiVo builds upon research and methods from:
  * **CelloCut:** Constructive Watertight Remeshing via Tetrahedral Cell Cuts (Xuan Yang et al.)
  * **FaithC / Faithful Contouring** (Yihao Luo et al.)
  * OpenVDB, CGAL, oneTBB, Eigen, and PyTorch.



## 🚀 SUPPORT MOSTAADTECH

### ❤️ Enjoying this project / workflow?

I’m **MostAadTech**, I create FREE ComfyUI workflows, local AI tools, 3D pipelines, and open-source projects.

If this project or workflow helped you, **please consider following me or supporting my work**. It helps me keep building, testing, and releasing more free tools and workflows.

---

## 💜 Support Me on Patreon

👉 **[Support MostAadTech on Patreon](https://www.patreon.com/cw/MostafaAwad/membership)**

Your support helps me spend more time developing **FREE AI tools, ComfyUI workflows, and 3D pipelines**.

---

## 🌐 Follow MostAadTech

* ▶️ **[YouTube](https://www.youtube.com/@MostAadTech)** — Tutorials, workflows & AI projects
* 📸 **[Instagram](https://www.instagram.com/mostaadtech/)** — Projects, updates & behind the scenes
* 𝕏 **[X / Twitter](https://x.com/MostAadTech)** — Updates, releases & experiments
* 💻 **[GitHub](https://github.com/Mstafa-awad)** — Open-source projects & code

---

### ⭐ One Follow Helps

**Follow • Star • Share • Support**

Every follow, GitHub star, share, and Patreon supporter helps me continue making **FREE tools for the AI community.**

**Thank you for supporting MostAadTech! ❤️**
