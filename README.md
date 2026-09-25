# WTiVo — WatertightVoxel (ComfyUI Node)

A high-performance, in-process watertight remeshing node for ComfyUI. WTiVo converts defective, non-manifold, or open 3D triangle meshes (such as raw outputs from AI 3D generators like TRELLIS) into dense, closed, manifold meshes using sparse voxel fields, tetrahedral cell cuts, CUDA graph optimization, and manifold contouring.

## 🎥 Watch the WTiVo Node in Action

**Want to see how WTiVo works? Watch the full video explanation here:**

[![WTiVo — WatertightVoxel ComfyUI Node](https://img.youtube.com/vi/QwILmqyjEow/maxresdefault.jpg)](https://www.youtube.com/watch?v=QwILmqyjEow&t=105s)

▶️ **[Watch the video on YouTube](https://www.youtube.com/watch?v=QwILmqyjEow&t=105s)**

---

## ⚡ Key Highlights & Changes

* **Native ComfyUI Custom Node:** Converted from a standalone CLI script into a native, in-process ComfyUI node (`WTiVo - Mesh Watertight`), executing directly on native `MESH` objects.
* **2K / High-Detail Reconstruction:** WTiVo can now process **2K meshes with significantly more geometric detail** by increasing the input resolution, final resolution, and proxy-point budget.
* **Higher Proxy-Point Quality:** The previous `12 million` proxy-point workflow can be increased to approximately **25 million proxy points** for high-detail 2K reconstruction.
* **More Proxy Points = More Detail:** Increasing the proxy-point budget gives the reconstruction more geometric information to work with. For detailed meshes, higher proxy-point counts can improve preservation of small features, edges, and surface detail.
* **16 GB+ VRAM Recommended for High Detail:** A GPU with **16 GB VRAM or more** is recommended when using 2K resolution together with approximately 25 million proxy points.
* **Prebuilt Backend (No Compilation Required):** Ships with precompiled CPython 3.12 / CUDA 12.8 `.pyd` native extensions. No MSVC, CMake, CGAL, or manual CUDA compilation steps (`Setup-Windows.cmd`) are needed for normal installation.
* **In-Memory Zero-IPC Pipeline:** Automatically unloads active upstream models before processing and passes native PyTorch tensors and NumPy arrays directly in memory—no temporary GLB/OBJ files or subprocess bridges.
* **CUDA-Accelerated Graph Cut:** Leverages multi-discharge Push-Relabel CUDA graph optimization and CGAL Delaunay tetrahedralization for fast watertight surface extraction.
* **Exact Edge Validation:** Performs topology validation using exact edge-degree checks to verify the resulting mesh topology.

---

## 🚀 New: 2K High-Detail Mesh Support

WTiVo is no longer limited to the previous 1536-resolution / 12-million-proxy-point workflow.

For **2K meshes containing a lot of geometric detail**, increase the reconstruction settings:

```text
input_res      = 2048
final_res      = 2048
proxy_points   = 25,000,000
```

### Recommended configurations

| Workflow       | Input Resolution | Final Resolution | Proxy Points | VRAM                    |
| :------------- | :--------------: | :--------------: | :----------: | :---------------------- |
| Standard       |      `1536`      |      `1536`      |     `12M`    | ~8 GB+                  |
| High Detail    |      `1536`      |      `1536`      |     `20M`    | Higher VRAM recommended |
| 2K High Detail |      `2048`      |      `2048`      |     `25M`    | **16 GB+ recommended**  |

These are practical starting points rather than hard limits.

### More proxy points can preserve more detail

The proxy-point budget is an important part of the reconstruction quality.

For example:

```text
12,000,000 proxy points
        ↓
Good for standard workflows

20,000,000 proxy points
        ↓
Higher geometric budget

25,000,000+ proxy points
        ↓
Designed for high-detail / 2K workflows
```

If your GPU has enough VRAM, increasing the number of proxy points can provide the reconstruction with more geometric information and can help preserve fine details.

**For GPUs with 16 GB VRAM or more, around 25 million proxy points is recommended for 2K high-detail workflows.**

> **Important:** More proxy points require more memory. If you run out of VRAM, reduce `proxy_points` first or lower the reconstruction resolution.

---

## 🖥️ GPU Build Compatibility

WTiVo includes prebuilt native `.pyd` extensions.

The `build` folder currently contains dedicated binaries for **RTX 50-series GPUs**.

### 🟢 RTX 50 Series

If you are using an RTX 50-series GPU:

**You can keep using the dedicated `.pyd` files already inside the `build` folder.**

No additional extraction is required.

There is **no performance difference** between keeping the dedicated RTX 50-series `.pyd` files and using the compatible extracted build.

---

### 🔵 Older NVIDIA GPUs

If your GPU is **older than the RTX 50 series**, the dedicated `.pyd` files currently inside `build` are not the correct binaries for your GPU.

You need to extract the compatible build package.

#### Steps

1. Open:

```text
WTiVo-WatertightVoxel-ComfyuiNode/build/
```

2. Locate the compatible compressed build package.

3. **Unzip/extract it directly into the `build` directory.**

4. Allow the compatible `.pyd` files to be placed in the build directory.

5. Restart ComfyUI.

The compatible build is intended for supported older NVIDIA GPUs.

### In short

```text
RTX 50 Series
│
├── Keep the dedicated .pyd files
└── No extraction required


Older than RTX 50 Series
│
├── Open build/
├── Extract the compatible build package
└── Use the extracted .pyd files
```

**You do not need to extract the compatibility package if you are using an RTX 50-series GPU.**

---

## 📋 System Requirements

* **OS:** Windows 10 / 11 x64
* **GPU:** NVIDIA CUDA-capable GPU
* **GPU Support:** RTX 20 / 30 / 40 / 50 Series, provided the correct compatible native build is used
* **VRAM:** Approximately **8 GB+** recommended for standard workflows
* **VRAM for 2K/high detail:** **16 GB+ recommended**
* **Environment:** ComfyUI with **Python 3.12 embedded**
* **PyTorch:** **PyTorch 2.8.0 with CUDA 12.8**

Example:

```text
torch-2.8.0+cu128-cp312-cp312-win_amd64.whl
```

This is typically located in:

```text
\ComfyUI-Easy-Install\python_embeded\
```

or your custom embedded Python environment.

### Memory recommendation

A larger reconstruction requires substantially more memory.

For example:

```text
1536 + 12M proxy points
```

uses considerably less memory than:

```text
2048 + 25M proxy points
```

For this reason, **16 GB VRAM or more is recommended for 2K/high-detail workflows**.

---

## 📦 Installation

### 1. Verify PyTorch

Ensure your embedded Python environment has:

```text
torch 2.8.0+cu128
```

For ComfyUI Easy Install environments:

```cmd
.\python_embeded\python.exe -m pip install torch-2.8.0+cu128-cp312-cp312-win_amd64.whl
```

---

### 2. Install Custom Node

Clone or extract this repository into your ComfyUI custom-nodes directory:

```cmd
cd ComfyUI\custom_nodes\
git clone https://github.com/Mstafa-awad/WTiVo-WatertightVoxel-ComfyuiNode.git
```

---

### 3. Select the Correct Native Build

#### RTX 50-series GPU

Keep the dedicated `.pyd` files already provided in:

```text
build/
```

No extraction is necessary.

#### Older NVIDIA GPU

Extract the compatible build package into:

```text
build/
```

so that the appropriate `.pyd` files are available to WTiVo.

---

### 4. Restart ComfyUI

Launch ComfyUI.

The node will be available under:

```text
3d/mesh/WTiVo
```

as:

**WTiVo - Mesh Watertight**

---

## ⚙️ Node Parameters

| Parameter               | Type      |   Default  | Description                                                                                                                                             |
| :---------------------- | :-------- | :--------: | :------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mesh`                  | **MESH**  |      —     | Input ComfyUI native mesh object, e.g. from `VaeDecodeShapeTrellis`.                                                                                    |
| `input_res`             | **INT**   |   `1536`   | Resolution for the initial thick UDF sparse field and graph-labeling pass. Increase to `2048` for 2K workflows.                                         |
| `final_res`             | **INT**   |   `1536`   | Resolution for the final signed OpenVDB field and FaithC reconstruction. Increase to `2048` for 2K workflows.                                           |
| `proxy_points`          | **INT**   | `12000000` | QEF proxy-point budget sent to CGAL for tetrahedralization. Around `25,000,000` is recommended for high-detail 2K meshes when enough VRAM is available. |
| `proxy_eps_scale`       | **FLOAT** |    `1.0`   | Inset scale for graph proxy geometry relative to resolution.                                                                                            |
| `proxy_feature_weight`  | **FLOAT** |    `1.5`   | Priority weight for preserving sharp corners, edges, and creases.                                                                                       |
| `lambda_fill`           | **FLOAT** |   `20.0`   | Regularization factor for tetrahedral cell-cut fill.                                                                                                    |
| `thin_iso_vox`          | **FLOAT** |    `0.0`   | Surface offset in voxel units for the final signed field.                                                                                               |
| `faithc_component_mode` | **ENUM**  |  `largest` | Component handling: `auto`, `keep_all`, or `largest`.                                                                                                   |
| `threads`               | **INT**   |  *Max CPU* | Number of logical CPU threads allocated for parallel operations.                                                                                        |

---

## 🔄 Workflow Integration

```text
[ VaeDecodeShapeTrellis ]
          │
          │ MESH
          ▼
[ WTiVo - Mesh Watertight ]
          │
          │ Watertight MESH
          ▼
[ Preview 3D / Save 3D / Exporter ]
```

### Standard workflow

```text
input_res      = 1536
final_res      = 1536
proxy_points   = 12,000,000
```

### 2K high-detail workflow

```text
input_res      = 2048
final_res      = 2048
proxy_points   = 25,000,000
```

For GPUs with **16 GB VRAM or more**, the 2K configuration provides a substantially larger reconstruction budget for detailed geometry.

> **Note:** WTiVo completely rebuilds topology to create a watertight surface. Input UVs, textures, and material maps are not preserved. Ensure batch size is set to `1`.

---

## 🧠 Quality vs Memory

WTiVo's reconstruction quality is affected by both **resolution** and **proxy-point density**.

A simplified way to think about it:

```text
Higher input resolution
          +
Higher final resolution
          +
More proxy points
          ↓
More geometric information
          ↓
Potentially more preserved detail
          ↓
Higher VRAM / RAM usage
```

For example:

```text
1536 / 1536 / 12M
```

is designed around a more moderate memory budget.

Whereas:

```text
2048 / 2048 / 25M
```

is designed for **high-detail 2K reconstruction**.

If your GPU has more available VRAM, increasing the proxy-point count can allow WTiVo to work with a larger geometric budget.

---

## ⚠️ Important Notes

* WTiVo completely rebuilds the mesh topology during watertight reconstruction.
* The resulting mesh is designed to be closed and manifold when reconstruction succeeds.
* Input UVs, textures, and material maps are **not preserved**.
* Use **batch size 1**.
* Higher resolutions and proxy-point counts require significantly more memory.
* **16 GB VRAM or more is recommended for 2K / ~25M proxy-point workflows.**
* If you encounter VRAM or memory limitations, reduce `proxy_points` first, then reduce `input_res` / `final_res`.
* More proxy points generally provide a larger geometric budget and can improve preservation of fine details, but the final result also depends on the input mesh and other reconstruction settings.
* Use the correct `.pyd` build for your GPU generation.

---

## 🔍 Topology Validation

WTiVo performs topology validation on the reconstructed mesh.

The backend checks edge groups to determine whether the resulting surface has:

```text
boundary edges = 0
non-manifold edges = 0
```

A successful watertight result is reported as:

```text
watertight=True
```

This makes WTiVo useful as a preprocessing stage for:

* AI-generated 3D assets
* mesh optimization
* low-poly conversion
* baking
* game-ready asset pipelines
* further remeshing
* 3D printing workflows

---

## ⚖️ License & Attribution

* **License:** GNU General Public License v3.0 or later (**GPL-3.0-or-later**).
* **Attribution:** WTiVo builds upon research and methods from:

  * **CelloCut:** Constructive Watertight Remeshing via Tetrahedral Cell Cuts (Xuan Yang et al.)
  * **FaithC / Faithful Contouring** (Yihao Luo et al.)
  * OpenVDB
  * CGAL
  * oneTBB
  * Eigen
  * PyTorch

---

## 🚀 SUPPORT MOSTAADTECH

### ❤️ Enjoying this project / workflow?

I'm **MostAadTech**, I create FREE ComfyUI workflows, local AI tools, 3D pipelines, and open-source projects.

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
