# MyoGait Clinical Gait Analysis Guide: Zero-to-Hero Manual

Welcome to the **MyoGait Clinical Gait Analysis Platform**! This guide is written specifically for clinical researchers, clinicians, fellows, and students with **zero prior programming or coding experience**.

By following these step-by-step instructions, you will learn how to open the project in Visual Studio Code (VS Code), set up the GPU-accelerated environment with one click, run the clinical dashboard, digitize patient Case Report Forms (CRFs), and generate complete biomechanical reports.

---

## Table of Contents
1. [Prerequisites: What You Need on Your Computer](#1-prerequisites-what-you-need-on-your-computer)
2. [Step 1: Opening the Project in VS Code](#2-step-1-opening-the-project-in-vs-code)
3. [Step 2: One-Step Automated Environment Setup](#3-step-2-one-step-automated-environment-setup)
4. [Step 3: Selecting the Python Interpreter in VS Code](#4-step-3-selecting-the-python-interpreter-in-vs-code)
5. [Step 4: Launching the Dashboard from VS Code](#5-step-4-launching-the-dashboard-from-vs-code)
6. [Hardware Guide & GPU Clearance (NVIDIA RTX 6000 24GB)](#6-hardware-guide--gpu-clearance)
7. [Step-by-Step Clinical Workflow in the Dashboard](#7-step-by-step-clinical-workflow-in-the-dashboard)
8. [Where Results Are Saved & Viewing Outputs](#8-where-results-are-saved--viewing-outputs)
9. [Beginner Troubleshooting & FAQ](#9-beginner-troubleshooting--faq)

---

## 1. Prerequisites: What You Need on Your Computer

Before starting, ensure the following three free tools are installed on your workstation:

1. **Visual Studio Code (VS Code)**: Your code editor and workspace manager.  
   👉 [Download VS Code for Windows/Mac](https://code.visualstudio.com/)
2. **Miniconda or Anaconda**: Manages Python and GPU packages securely.  
   👉 [Download Miniconda for Windows (64-bit)](https://docs.anaconda.com/miniconda/)
   *(During Anaconda/Miniconda installation, check the box: "Add Miniconda3 to my PATH environment variable" if prompted, or use the "Anaconda Prompt").*
3. **NVIDIA GPU Drivers** *(Already installed if you have an NVIDIA Quadro RTX 6000 / RTX 3000 / RTX 4000 PC)*.

---

## 2. Step 1: Opening the Project in VS Code

1. **Open Visual Studio Code**.
2. Go to the top menu bar: click **`File`** > **`Open Folder...`** (or press `Ctrl + K`, then `Ctrl + O` on Windows; `Cmd + O` on Mac).
3. Navigate to where the `myogait` project is located on your computer, select the main **`myogait`** folder, and click **Select Folder**.
4. If VS Code shows a pop-up saying *"Do you trust the authors of the files in this folder?"*, click **"Yes, I trust the authors"**.

### Tour of VS Code for Beginners:
- **Left Sidebar (File Explorer)**: Lists all folders and files (`myogait/`, `dcm_dashboard.py`, `Output/`, etc.).
- **Center Area (Editor)**: Where file contents are displayed.
- **Bottom Area (Integrated Terminal)**: Where you run commands. (You can toggle it open or closed at any time by pressing **`Ctrl + ~`** or going to **`Terminal` > `New Terminal`**).

---

## 3. Step 2: One-Step Automated Environment Setup

We provide an automated batch script that configures Python, CUDA GPU drivers, PyTorch, and all AI pose models in a single click.

### In VS Code:
1. Open the integrated terminal: Click **`Terminal`** > **`New Terminal`** at the top menu.
2. In the terminal window at the bottom, type the following command and press **Enter**:

   **On Windows (PC with NVIDIA GPU):**
   ```cmd
   .\install_windows_gpu.bat
   ```

   **On macOS / Linux:**
   ```bash
   chmod +x ./install_linux_gpu.sh
   ./install_linux_gpu.sh
   ```

3. **What the script does automatically:**
   - Creates a dedicated Conda environment named **`dcm-gait`** (Python 3.11).
   - Installs **PyTorch with CUDA 12.4** GPU acceleration.
   - Installs all MyoGait AI pose backends, Streamlit dashboard, OpenSim exporters, and Excel/PDF generators.
   - Tests your GPU and prints a verification box:
     ```text
     --------------------------------------------------
     PyTorch Version: 2.x
     CUDA Available: True
     GPU Name: NVIDIA Quadro RTX 6000
     VRAM (GB): 24.0
     --------------------------------------------------
     ```

---

## 4. Step 3: Selecting the Python Interpreter in VS Code

To make sure VS Code uses the newly created `dcm-gait` environment:

1. Press **`Ctrl + Shift + P`** (Windows) or **`Cmd + Shift + P`** (Mac) to open the Command Palette.
2. Type: **`Python: Select Interpreter`** and press **Enter**.
3. Select **`Python 3.11 (dcm-gait: conda)`** from the list.
4. If prompted to install the Microsoft Python Extension, click **Install**.

---

## 5. Step 4: Launching the Dashboard from VS Code

### Method 1: The One-Command Launch (Inside VS Code Terminal)
In the VS Code Terminal, ensure the `dcm-gait` environment is active (you will see `(dcm-gait)` on the left side of the prompt), then type:

```bash
streamlit run dcm_dashboard.py
```

### Method 2: The One-Click Batch Launcher (Windows Explorer)
Alternatively, navigate to the `myogait` folder in Windows File Explorer and double-click **`run_dashboard.bat`**.

---

### What Happens Next:
- A local server will start.
- Your default web browser (Chrome, Edge, Firefox, Safari) will automatically pop open to **`http://localhost:8501`**.
- You will see the **MyoGait DCM Clinical Dashboard** with two tabs:
  1. **🏃 Pipeline Analysis**
  2. **📂 Clinical Database (CRF)**

> 💡 **To stop the dashboard**: Click inside the VS Code Terminal and press **`Ctrl + C`**.

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
   - **Recording Plane**:
     - `sagittal` (side-view video)
     - `coronal` (front or back-view video)
   - **Pose Model**:
     - On RTX 6000: Select **`sapiens2-ultra`**, **`sapiens2-top`**, or **`vitpose-huge`** for maximum research accuracy.
     - For fast real-time testing: Select **`mediapipe`** or **`yolo`**.
   - **Gait Event Method**: Keep as `zeni` (standard) or choose `crossing` / `velocity`.
   - **Frame Quality Preset**:
     - `Distant subject` (Default: adaptive visibility floor, recommended for wide capture angles).
     - `Standard` (Stricter gating, recommended for close well-framed recordings).
   - **Visible Side (Sagittal Only)**:
     - `both`: Keeps both legs.
     - `left` or `right`: If the patient walked such that one leg was obscured from camera view, pick the visible leg. The pipeline will automatically scrub noisy occluded data.
3. In **Video Selection**:
   - Select **File Upload** to drag-and-drop your MP4/MOV video, or **Local File Path** to paste the exact video location.
4. Click the large blue **`Run Clinical Analysis`** button.
5. A spinner will show the 7-step analysis:
   `Extracting landmarks` ➔ `Filtering pose data` ➔ `Computing joint angles` ➔ `Perspective correction` ➔ `Gait cycle segmentation` ➔ `Symmetry & GPS scores` ➔ `Generating PDF/Excel/OpenSim`.

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

### Viewing in VS Code:
1. In the VS Code left sidebar, expand `Output` > `Patients` > `[Patient_ID]` > `[Timestamp]`.
2. Right-click on `dcm_gait_report.pdf` or `summary.png` and click **Reveal in File Explorer** (Windows) or **Reveal in Finder** (Mac) to open and print.

---

## 9. Beginner Troubleshooting & FAQ

### Q1: I get `'conda' is not recognized as an internal or external command` on Windows.
- **Cause**: Miniconda was installed without adding the command line tools to your system path.
- **Fix**: Open the **Anaconda Prompt** from your Windows Start Menu, navigate to the folder with `cd "C:\path\to\myogait"`, and run `install_windows_gpu.bat`.

### Q2: The terminal says `Port 8501 is already in use`.
- **Cause**: An earlier session of the dashboard is still running in the background.
- **Fix**: In the VS Code Terminal, press **`Ctrl + C`** to stop it, or run:
  ```bash
  streamlit run dcm_dashboard.py --server.port 8502
  ```

### Q3: How do I update the code if new changes are pushed to GitHub?
- In the VS Code Terminal, simply type:
  ```bash
  git pull origin master
  ```

### Q4: I ran a video, but GPS (Gait Profile Score) says `N/A`.
- **Cause**: GPS is calculated for Sagittal (side-view) recordings using bilateral hip, knee, and ankle kinematics. If coronal plane or single-side scrubbing is enabled, individual joint kinematics are reported instead of bilateral composite GPS.

---
*For questions, clinical collaboration, or bug reports, please open an issue on the repository or contact the research team.*
