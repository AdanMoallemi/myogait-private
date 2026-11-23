# MyoGait Clinical Gait Analysis Guide: Zero-to-Hero Manual

Welcome to the **MyoGait Clinical Gait Analysis Platform**! This guide is written specifically for clinical researchers, clinicians, fellows, and students with **zero prior programming or coding experience**.

By following these step-by-step instructions, you will learn how to install the software on a Windows PC with one double-click, run the clinical dashboard, digitize patient Case Report Forms (CRFs), and generate complete biomechanical reports.

---

## Table of Contents
1. [Prerequisites: What You Need on Your Computer](#1-prerequisites-what-you-need-on-your-computer)
2. [Step 1: Getting the Project onto the PC](#2-step-1-getting-the-project-onto-the-pc)
3. [Step 2: One-Step Automated Setup](#3-step-2-one-step-automated-setup)
4. [Step 3: Launching the Dashboard](#4-step-3-launching-the-dashboard)
5. [Step 4: Updating to a New Version](#5-step-4-updating-to-a-new-version)
6. [Hardware Guide & GPU Clearance (NVIDIA RTX 6000 24GB)](#6-hardware-guide--gpu-clearance)
7. [Step-by-Step Clinical Workflow in the Dashboard](#7-step-by-step-clinical-workflow-in-the-dashboard)
8. [Where Results Are Saved & Viewing Outputs](#8-where-results-are-saved--viewing-outputs)
9. [Beginner Troubleshooting & FAQ](#9-beginner-troubleshooting--faq)

---

## 1. Prerequisites: What You Need on Your Computer

- A **Windows 10 or 11** PC with an internet connection. Remote Desktop is fine: the dashboard
  runs in the PC's web browser.
- An **NVIDIA graphics card** with its driver installed (recommended). Check by opening
  *Command Prompt* and typing `nvidia-smi`; if that prints a table with the GPU name, you are
  ready. If not, install the driver from [nvidia.com/drivers](https://www.nvidia.com/drivers).
  Without a GPU everything still works on the CPU with the `mediapipe` pose model, only slower
  and less accurately.
- About **15 GB of free disk space** on the drive where you put the project.
- Your **GitHub login** (the repository is private).

Nothing else: you do **not** need to install Python, Conda or VS Code. The installer downloads
its own copy of Python and keeps everything (packages, caches, model weights) inside the project
folder, on the same drive.

---

## 2. Step 1: Getting the Project onto the PC

Use a **short folder path without spaces**, for example `C:\myogait-private` or
`D:\myogait-private` (Windows limits file paths to 260 characters, and some packages have deep
folders).

**Option A — Git (recommended: one-click updates later).** Open *PowerShell* and run:

```powershell
winget install --id Git.Git -e
```

Close and reopen PowerShell, then:

```powershell
git clone https://github.com/AdanMoallemi/myogait-private.git C:\myogait-private
```

A browser window asks you to sign in to GitHub the first time.

**Option B — ZIP (no Git).** In the PC's browser, sign in to GitHub, open the repository, click
**Code ▸ Download ZIP**, and extract it to `C:\myogait-private`. (A ZIP copy cannot update
itself; for a new version, download the ZIP again.)

---

## 3. Step 2: One-Step Automated Setup

In the project folder, **double-click `install_windows_gpu.bat`**.

If Windows shows *"Windows protected your PC"*, click **More info ▸ Run anyway** (this happens
for files downloaded from the internet).

The installer runs 7 steps on its own, about 10-20 minutes the first time:

1. Detects the NVIDIA GPU and driver.
2. Downloads `uv`, a small tool that installs Python and packages quickly.
3. Creates a private Python 3.12 environment in the `.venv` folder.
4. Installs PyTorch built for your GPU driver (CUDA 12.8, 12.6 or 11.8, chosen automatically,
   then tested on the GPU; it falls back automatically if a build does not work).
5. Installs myogait and the **exact package versions** of `requirements\dcm-lock.txt`, so every
   installation computes identical numbers.
6. Downloads the **Sapiens 2 Quick** pose model (myogait's default) and prepares it for the GPU.
7. Creates a **"MyoGait DCM Dashboard"** shortcut on the desktop.

Everything is logged to `install_log.txt` in the project folder. Running the installer again is
always safe; it reuses what is already installed. If a step fails, the message says what to fix;
then double-click the installer again.

Options (from PowerShell in the project folder), rarely needed:
`.\install_windows_gpu.bat -Cuda cu126` forces a PyTorch build;
`-SkipModels` skips the model download.

---

## 4. Step 3: Launching the Dashboard

**Double-click "MyoGait DCM Dashboard" on the desktop** (or `run_dashboard.bat` in the project
folder). A black window opens and the dashboard appears in the browser at
`http://127.0.0.1:8501`. **Keep the black window open** while you work; close it to stop the
dashboard.

The dashboard is only reachable from this PC (including over Remote Desktop), not from the rest
of the network, so no firewall prompt appears. Videos up to 4 GB can be uploaded.

---

## 5. Step 4: Updating to a New Version

- **Git copy (Option A):** double-click **`update_windows.bat`**. It downloads the new version
  and refreshes the packages (fast: only what changed).
- **ZIP copy (Option B):** download the ZIP again into a new folder and run
  `install_windows_gpu.bat` there. Copy your `Output` folder (patients and results) across.

Your patient database and all results live in the `Output` folder, which updates never touch.

---

## 6. Hardware Guide & GPU Clearance

The dashboard automatically checks your hardware on startup and displays your GPU status card.

### Your Workstation: NVIDIA Quadro RTX 6000 (24 GB VRAM)
Your machine is classified as a **Tier 1 (Workstation Grade)** system:
- **Massive 24 GB VRAM**: Full clearance to run the largest foundation models (**Sapiens 2 Ultra 5B parameters**, **Sapiens 2 Top 1B**, **ViTPose Huge**) without any risk of Out-Of-Memory (OOM) errors.

### Hardware Tier & Model Matrix:

| Workstation Tier | Detected VRAM | Recommended Models in Dashboard | Clinical Clearance & Waivers |
| :--- | :--- | :--- | :--- |
| **Tier 1: Workstation GPU**<br>*(e.g., Quadro RTX 6000 24GB, RTX 4090)* | **24 GB+** | `sapiens2-ultra` (5B)<br>`sapiens2-top` (1B)<br>`vitpose-huge`<br>`sapiens-top`<br>`yolo`, `mediapipe` | **Full Clearance**: Can run all 5-billion parameter foundation models and 4K footage simultaneously with maximum precision. |
| **Tier 2: High-End Consumer GPU**<br>*(e.g., RTX 3070/4070, RTX 3080 10GB/12GB)* | **8 GB – 12 GB** | `sapiens2-mid` (0.8B)<br>`sapiens2-quick` (0.4B)<br>`vitpose-large`<br>`yolo`, `mediapipe` | **Approved for Standard Models**: Sapiens 2 Mid and ViTPose Large run smoothly. Avoid `sapiens2-ultra` on long videos. |
| **Tier 3: Entry / Laptop GPU**<br>*(e.g., GTX 1660, RTX 3050, 4GB/6GB VRAM)* | **4 GB – 6 GB** | `sapiens2-quick` (0.4B)<br>`vitpose` (base)<br>`yolo`<br>`rtmw`<br>`mediapipe` | **Approved for Lightweight Models**: Fast and robust. Avoid 1B+ parameter models. |
| **Tier 4: CPU Mode / Standard Laptop**<br>*(Intel/AMD CPU only, no CUDA)* | **System RAM** | `mediapipe`<br>`yolo` | **CPU Safe**: `mediapipe` provides real-time CPU tracking. Transformer models (`sapiens`, `vitpose`) will run slowly without GPU. |

---

## 7. Step-by-Step Clinical Workflow in the Dashboard

### Part A: The Clinical Database (CRF) Tab
> ⚠️ **Always start here for any new patient!**

1. Click on **📂 Clinical Database (CRF)** at the top.
2. Under **Select Patient Profile**, choose **`+ Create New Patient`**.
3. Type the Patient ID (e.g., `DCM-001` or `PAT-102`).
4. Fill out the Case Report Form sections:
   - **Demographics**: Enter Age, Sex, Affected Side, and **Height (cm)**.  
     *(Height is critical: MyoGait uses it to scale joint angles and linear walking distance in metres).*
   - **Surgical & Medical History**: Record symptom duration, prior decompressions, myelopathy status.
   - **mJOA Assessment**: Fill out Upper/Lower motor, sensory, and bladder scores (the mJOA severity score calculates automatically).
   - **10MWT Results**: Enter preferred and fast walk speeds.
   - **Video File Log**: Paste paths to your raw camera videos for record-keeping.
5. **CRITICAL STEP**: Click the **`💾 Save Patient Profile`** button at the bottom of the page.

---

### Part B: The Pipeline Analysis Tab

1. Switch to the **🏃 Pipeline Analysis** tab.
2. In **Pipeline Setup**:
   - **Experimenter**: Select your name (or use `+ Manage List` to add yourself).
   - **Select Patient**: Choose the patient you just created (e.g., `DCM-001`). The dashboard will confirm: `Loaded DCM-001: Height = 1.75m`.
   - **Analysis Settings**: every setting is listed, grouped in panels, with an ⓘ explanation
     (hover it) and its myogait default. **All defaults are myogait's own validated defaults**, so
     a run with untouched settings is identical to `myogait.run_pipeline`. A green banner confirms
     this; if you change anything that affects the numbers, a yellow banner lists it, and the run's
     README.md and run_metadata.json record it. **↺ Reset to myogait defaults** restores everything.
     Settings you are most likely to touch:
     - **Pose model** (default `sapiens2-quick`, myogait's validated model; needs the GPU).
       Process every patient and session with the same model.
     - **Longest stride kept** (default 1.6 s). Strides slower than this are discarded. For a
       slow, severe DCM walker whose cycles are reported as dropped, raise it (e.g. 2.5 s).
     - **Frame quality preset**: `Standard (myogait default)`, or `Distant subject` when the
       patient is far from the camera and too many frames are discarded.
     - **Side to analyse**: `both` (default) or the leg nearest the camera; the run README says
       which leg faced the camera. Symmetry needs `both`.
     - **Recording plane**: `sagittal` (side view) or `coronal` (front/back view). It is recorded
       in the outputs; for coronal videos also untick *Normalise walking direction*.
     - **Optional corrections** (perspective, drift removal, median spike filter, healthy-adult
       bias correction) are off by default because they are not part of myogait's validated
       recipe; the bias correction should stay off for DCM patients.
   - **Left/right labels** always refer to the patient's true sides, in the data, the report and
     the labelled videos.
3. In **Video Selection**:
   - Select **File Upload** to drag-and-drop your MP4/MOV video, or **Local File Path** to paste the exact video location.
4. Click the large blue **`Run Clinical Analysis`** button.
5. A spinner runs the analysis: pose extraction ➔ myogait calculations (filtering, angles, gait
   events, cycles, statistics, scores) ➔ PDF / Excel / CSV / OpenSim / plots / videos. Any
   myogait quality warning (few cycles, out-of-plane distortion, dropped strides) is shown
   above the results.

---

## 8. Where Results Are Saved & Viewing Outputs

Once analysis finishes, all files are permanently organized by Patient ID and timestamp inside the **`Output/`** folder:

```text
Output/
└── Patients/
    └── DCM-001/
        └── 2026-08-20_11-35-17_gait_trial_01/
            ├── README.md                      <-- Permanent clinical audit trail
            ├── run_metadata.json              <-- Machine-readable run parameters
            ├── profile.json                   <-- Patient CRF snapshot
            ├── report/
            │   └── dcm_gait_report.pdf        <-- Formal Clinical PDF Report
            ├── data/
            │   ├── dcm_gait_analysis.xlsx     <-- Multi-sheet Excel workbook
            │   └── csv/                       <-- 14 Raw CSV kinematic tables
            ├── opensim/
            │   ├── kinematics.mot             <-- OpenSim joint motion
            │   └── markers.trc                <-- OpenSim virtual marker trajectories
            └── plots/
                ├── summary.png                <-- 6-panel summary dashboard
                ├── angles.png                 <-- Joint angle timeseries
                ├── cycles.png                 <-- Normalized gait cycle waveforms
                └── skeleton_overlay.mp4       <-- Video with AI pose overlay
```

### Viewing the files:
1. In Windows File Explorer, open the project folder, then `Output` > `Patients` > `[Patient_ID]` > `[Timestamp]`.
2. Double-click `report\dcm_gait_report.pdf` or `plots\summary.png` to open and print. The dashboard also offers download buttons for the PDF, Excel workbook and stick-figure video.

---

## 9. Beginner Troubleshooting & FAQ

### Q1: The installer stopped with "SETUP FAILED".
- Read the line above the message; `install_log.txt` in the project folder has every detail.
- Most common causes: no internet connection (github.com, pypi.org and download.pytorch.org
  must be reachable), a folder path that is too long (move the project to `C:\myogait-private`),
  or not enough disk space. Fix it and double-click `install_windows_gpu.bat` again.

### Q2: The dashboard window says `Port 8501 is already in use`.
- **Cause**: an earlier dashboard is still running (another black window is open).
- **Fix**: close the other window, or open the dashboard that is already running at
  `http://127.0.0.1:8501`.

### Q3: The installer says it fell back to CPU, but the PC has an NVIDIA GPU.
- Update the NVIDIA driver from [nvidia.com/drivers](https://www.nvidia.com/drivers), restart,
  and run `install_windows_gpu.bat` again; it picks the matching PyTorch build automatically.

### Q4: I ran a video, but GPS (Gait Profile Score) says `N/A`.
- **Cause**: no gait cycle passed myogait's cycle filters (or one side had too few heel strikes).
  The warnings above the results say why. For a slow walker, raise **Longest stride kept**
  (Gait events & cycles panel) and run again.

---
*For questions, clinical collaboration, or bug reports, please open an issue on the repository or contact the research team.*
