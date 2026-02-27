# Plan: Fingerprint Recognition — ORB + Adaptive Threshold + Lowe's Ratio Test

## Context
Academic fingerprint recognition system. Dataset: 80 grayscale TIFF images
(300×300 px), 10 subjects (IDs 101–110), 8 samples each.
Filename convention: `{subject_id}_{sample_id}.tif`

---

## Files

| File | Purpose |
|---|---|
| `fingerprint_recognition.py` | Main recognition script |
| `spec_finger_recognition.md` | Technical specification |
| `plan_fingerprint.md` | This file — implementation roadmap |

---

## Pipeline (in order)

### Step 1 — Dataset Loading
- Scan `fingerprint_images/*.tif`
- Parse `subject_id` and `sample_id` from filename
- Build list of `(path, subject_id, sample_id)` tuples

### Step 2 — Query Selection
- Pick one random entry from the dataset list
- Record its `subject_id` as the ground-truth owner

### Step 3 — Preprocessing (per image)
- Read as grayscale
- Apply `cv2.adaptiveThreshold` (Gaussian, block=11, C=2) to sharpen
  ridge/valley contrast before keypoint detection

### Step 4 — Feature Extraction (ORB)
- `cv2.ORB_create(nfeatures=500)`
- `detectAndCompute` on the preprocessed image
- Returns keypoints + binary descriptor matrix

### Step 5 — Matching with Lowe's Ratio Test
- `BFMatcher(cv2.NORM_HAMMING, crossCheck=False)`
- `knnMatch(query_desc, candidate_desc, k=2)`
- Keep match `m` if `m.distance < 0.75 * n.distance`
- Count surviving good matches

### Step 6 — Score Aggregation
- Sum good-match counts per `subject_id` (excluding query image itself)
- Subject with highest aggregate score → predicted owner

### Step 7 — Reporting
- Elapsed wall-clock time
- Top-10 individual match table
- Subject-level score table
- Predicted vs. true owner + correctness flag

---

## Tunable Constants

| Constant | Default | Note |
|---|---|---|
| `ORB_NFEATURES` | 500 | Increase for more keypoints (slower) |
| `RATIO_TEST` | 0.75 | Lowe's original value |
| `ADAPTIVE_BLOCK_SIZE` | 11 | Must be odd |
| `ADAPTIVE_C` | 2 | Subtracted from mean; adjust for contrast |

---

## Verification
```bash
cd fingerpirnt_detection
python fingerprint_recognition.py
```
- Output must show query filename, match table, predicted and true owner.
- Predicted == true owner for the majority of random runs.
- Elapsed time expected < 5 s on a modern CPU.
- Re-run multiple times to confirm consistency.

---

## Resuming This Plan
If returning to this task mid-implementation, check which of the three files
above exist and resume from the first missing one.
