# Changelog

All notable changes to the WTiVo (WatertightVoxel Optimizer) ComfyUI node will be documented in this file.

## [1.2.0] - 2026-09-19

### 🚨 Critical Fixes
- **Resolved `MemoryError: bad allocation` Crashes:** Fixed a critical stability issue where running WTiVo multiple times in a row, or using it in heavy Multi-View workflows, would crash ComfyUI.
- **Fixed Virtual Memory Hoarding:** Addressed an issue where heavy C++ math libraries (CGAL and OpenVDB) would reserve up to ~50GB of Windows Virtual Memory (Commit Charge) and refuse to release it back to the OS. 
- **Eliminated Native Heap Fragmentation:** Prevented the Windows C++ runtime from exhausting the system's pagefile limits on consecutive executions.

### ⚙️ Architecture & Performance Changes
- **Subprocess Isolation Model:** Transitioned the heavy native mesh generation from an "in-process" execution model to an **Isolated Subprocess** architecture. 
  - *How it works:* The heavy C++ lifting now runs in a hidden background worker process. The moment the watertight mesh is generated, this worker terminates. Windows instantly and completely wipes all reserved virtual memory and fragmented heaps, guaranteeing a clean slate for the next run.
- **Real-Time Log Streaming:** Restored full visibility into the background worker. Progress bars, memory snapshots, and timing metrics now stream directly to the ComfyUI console in real-time.
- **NPY Bridge Implementation:** Re-implemented a secure, high-speed `.npy` array bridge to pass mesh data between ComfyUI and the isolated worker without disk I/O bottlenecks or serialization overhead.

### ✨ Quality & Compatibility
- **Zero Quality Loss:** The 12,000,000 proxy point budget, FaithC finalizer, and watertight auditing remain completely untouched. Output quality is mathematically identical to previous versions.
- **16GB RAM Systems Fully Supported:** Confirmed that physical RAM usage remains safely under 10GB during execution. The fix ensures that systems with 16GB of physical RAM can now run complex, multi-view workflows indefinitely without needing to restart ComfyUI.

### 🗑️ Removed
- Removed the flawed in-process memory retention logic that caused the "stair-stepping" memory leak across multiple prompt executions.
