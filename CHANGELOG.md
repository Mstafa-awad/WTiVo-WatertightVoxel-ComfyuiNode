# Changelog

All notable changes to the LODTailor Bake Forger project will be documented in this file.

## [v1.1.0] - 2026-09-19

### 🚀 New Features
- **Full PBR Metallic Baking Support**: The node now fully supports baking the Metallic channel, completing the standard PBR Metal-Roughness texture set required for game engines (Unreal Engine, Unity, etc.).
- **New Node Inputs**:
  - `bake_metallic_texture` (Boolean): Toggle to enable/disable metallic baking (Default: True).
  - `metallic_resolution` (Integer): Set the resolution for the metallic map (Default: 8192).
  - `material_metallic` (Float): Fallback fill value for uncovered UV texels (Default: 0.0).
- **New Node Output**: Added a dedicated `metallic` IMAGE output socket to the ComfyUI node.

### 🛠️ Technical Improvements & Under-the-Hood Fixes
- **The "Emission Swap" Trick**: Because Blender Cycles lacks a native `METALLIC` bake pass, the background script now uses a clever workaround. It temporarily reroutes the high-poly's Metallic input into an Emission shader, runs a standard `EMIT` bake pass to capture the data, and then flawlessly restores the original material nodes.
- **Correct Color Space & Buffer**: The metallic texture is now allocated with a 32-bit float buffer and "Non-Color" color space (matching Roughness and Normal maps) to prevent gamma clipping and ensure raw data accuracy.
- **Low-Poly Material Wiring**: The baked metallic map is now automatically linked directly to the `Metallic` input socket of the low-poly's Principled BSDF shader inside the exported GLB.
- **Coverage & Fallback Handling**: Updated the internal semantic map and texel-filling logic to correctly identify and fill the Metallic channel during two-pass fallback bakes.

### 📦 Export & Pipeline Updates
- The metallic map is now saved as `<model_name>_metallic.png` in the temporary output directory.
- The metallic PNG is loaded as a PyTorch tensor and correctly returned to the ComfyUI workflow for downstream masking or previewing.
- Updated `result.json` payload to include the `metallic_png` file path.