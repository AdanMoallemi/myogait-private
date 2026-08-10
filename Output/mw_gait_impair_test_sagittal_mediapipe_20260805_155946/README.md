# DCM Gait Analysis Run Report

**Run Timestamp**: 2026-08-05 16:00:05  
**Subject ID**: `mw_gait_impair_test`  
**Subject Height**: `1.61 m`  
**Video File**: `22-JUL-2026_pilot_MW_gait_impairement_right.MP4`  
**Recording Plane**: `Sagittal`  
**Pose Model**: `mediapipe`  
**Execution Time**: `19.2 s`  

---

## 1. Executive Clinical Summary

| Metric | Value | Reference / Notes |
| :--- | :--- | :--- |
| **Cadence** | `81.8` steps/min | Step rate |
| **Stride Time** | `1.466 ± 0.067` s | Mean ± SD |
| **Left Stance Phase** | `68.6` % | % Gait cycle |
| **Right Stance Phase** | `60.5` % | % Gait cycle |
| **Overall Symmetry Index (SI)** | `18.5` % | Normal < 10% |
| **Gait Profile Score (GPS)** | `None` ° | Lower = closer to normative |
| **Sagittal Deviation Index (SDI)** | `{'gdi_2d_left': 0.0, 'gdi_2d_right': 0.0, 'gdi_2d_overall': 0.0, 'note': 'myogait Sagittal Deviation Index (SDI): a 2D sagittal-plane deviation index using 4 variables (hip, knee, ankle, trunk). Normal gait scores ~100; pathological gait scores below 100. This is NOT the GDI of Schwartz & Rozumalski (2008). Use for screening only.'}` | Normal ~ 100 |

### Pathological Gait Flags
- ⚠️ **Asymmetry knee: SI=30.6% (>20%)**
- ⚠️ **Asymmetry ankle: SI=30.6% (>20%)**

---

## 2. Pipeline Settings & Processing Configuration

- **Pose Estimation Backend**: `mediapipe`
- **Spike Removal Filter**: 1D Median Filter (`kernel_size=5`)
- **Kinematic Denoising Filter**: Low-pass 2nd-order zero-phase Butterworth filter (`cutoff=5.0 Hz`)
- **Joint Angle Reference**: `sagittal_vertical_axis` (Davis et al. vertical reference method)
- **Perspective Correction**: `Enabled` (Zero-parameter physics-based cos α foreshortening fix)
- **Linear Detrending**: `Enabled` (Removes camera-to-subject walk-along distance drift on 5m walkway)
- **Empirical Healthy Bias Corrections**: `Bypassed` (**Preserves DCM pathological signatures**, e.g., foot drop, stiff knee)
- **Gait Event Detection**: `zeni` algorithm (Zeni et al. foot-to-pelvis displacement/velocity thresholding)

---

## 3. Output Folder Structure

```text
mw_gait_impair_test_sagittal_mediapipe_20260805_155946/
├── README.md                      <- This clinical & technical run summary
├── run_metadata.json              <- Machine-readable exact run configuration & stats
├── report/
│   └── dcm_gait_report.pdf        <- Multi-page Clinical PDF Report
├── data/
│   ├── dcm_gait_data.json         <- Full raw/normalized pivot JSON landmark database
│   ├── dcm_gait_analysis.xlsx     <- Multi-tab Excel workbook with stats & cycle curves
│   └── csv/                       <- Raw & cycle time-series CSV data tables
│       ├── raw_angles.csv
│       ├── cycles_left.csv
│       ├── cycles_right.csv
│       └── summary_stats.csv
├── opensim/
│   ├── kinematics.mot             <- OpenSim joint angle kinematics (.mot format)
│   └── markers.trc                <- OpenSim 3D marker trajectory data (.trc format)
└── plots/
    ├── summary.png                <- Full clinical summary dashboard
    ├── angles.png                 <- Joint angle time-series plot
    ├── events.png                 <- Event detection plot (heel-strike & toe-off)
    ├── cycles_left.png            <- Normalized left gait cycle curves (0-100%)
    └── cycles_right.png           <- Normalized right gait cycle curves (0-100%)
```

---
*Generated automatically by myogait DCM Clinical Gait Analysis Pipeline v0.6.0.*
