# Specification: Fingerprint Recognition System
### `fingerprint_recognition.py`

---

## 1. Overview

A command-line Python script that performs **automatic fingerprint recognition**
using classical computer-vision techniques. It selects a random fingerprint
image from a dataset, compares it against all others, and identifies which
subject (person) the fingerprint belongs to based on descriptor matching.

---

## 2. Dataset

| Property | Value |
|---|---|
| Location | `fingerprint_images/` (relative to script) |
| Format | TIFF (`.tif`), grayscale, 300 × 300 px |
| Subjects | 10 (IDs: 101 – 110) |
| Samples per subject | 8 |
| Total images | 80 |
| Naming convention | `{subject_id}_{sample_id}.tif` |

The subject ID is the ground-truth identity. Any two images that share a
subject ID are different impressions of the **same** finger.

---

## 3. Technologies & Libraries

| Library | Role |
|---|---|
| `opencv-python` (contrib) | Image I/O, preprocessing, ORB, BFMatcher |
| `numpy` | Array operations (descriptor matrices) |
| `os` | Filesystem traversal for dataset loading |
| `random` | Uniform random query image selection |
| `time` | Wall-clock elapsed time measurement |

> **Install:** `pip install opencv-contrib-python numpy`
> The `opencv-contrib-python` package is required (not the standard
> `opencv-python`) because ORB's `detectAndCompute` needs `cv2.ORB_create`.

---

## 4. Architecture

The script is structured as a **single-module pipeline** with five logical
sections, each implemented as a standalone function plus a `main()` entry point.

```
load_dataset()
      │
      ▼
random query selection
      │
      ├──► preprocess(query)  ──► extract_features(query)
      │
      ▼  (for each other image)
preprocess(candidate) ──► extract_features(candidate)
                                      │
                                      ▼
                          count_good_matches(query_desc, cand_desc)
                                      │
                                      ▼
                          subject_scores[subject_id] += good_matches
      │
      ▼
argmax(subject_scores) → predicted_subject
      │
      ▼
Report (table + timing + correctness)
```

---

## 5. Processing Steps in Detail

### 5.1 Dataset Loading — `load_dataset(images_dir)`
- Lists all `.tif` files in `images_dir` (sorted alphabetically for
  deterministic ordering).
- Splits the stem on `_` to extract `subject_id` and `sample_id`.
- Returns `list[(abs_path, subject_id, sample_id)]`.

### 5.2 Query Selection
- `random.choice(dataset)` picks one entry uniformly at random.
- The query's `subject_id` is the **ground-truth label** used to evaluate
  correctness at the end.

### 5.3 Preprocessing — `preprocess(gray_img)` — Adaptive Thresholding
- **Why**: Global thresholding fails when fingerprint images have uneven ink
  density or lighting. Adaptive thresholding computes a local threshold per
  pixel neighbourhood.
- **How**: `cv2.adaptiveThreshold` with `ADAPTIVE_THRESH_GAUSSIAN_C` uses a
  Gaussian-weighted sum of the `ADAPTIVE_BLOCK × ADAPTIVE_BLOCK` neighbourhood
  as the local mean, then subtracts constant `ADAPTIVE_C`.
  Result: binary image where ridge pixels are 0 and valley pixels are 255
  (or vice versa), enhancing the structural pattern for keypoint detection.
- **Parameters**: block size = 11 (covers ~3 ridge widths in 300 px images),
  C = 2 (prevents noise pixels from flipping).

### 5.4 Feature Extraction — `extract_features(preprocessed_img)`

#### ORB — Oriented FAST and Rotated BRIEF
| Component | Description |
|---|---|
| FAST | Corner detector using a circular pixel ring test; very fast |
| Harris score | Used to rank and retain the top `ORB_NFEATURES` keypoints |
| Intensity centroid | Provides orientation estimate for each keypoint |
| BRIEF | Binary descriptor: 256-bit string from random pixel-pair comparisons |
| Rotation invariance | Descriptors are computed in the keypoint's rotated frame |

- **Why ORB for fingerprints**: Ridge endings and bifurcations are natural
  corner-like structures detected reliably by FAST. Binary descriptors make
  matching very fast via Hamming distance.
- Returns `(keypoints, descriptors)` — descriptors is an `N×32` `uint8` matrix
  (each row is a 256-bit / 32-byte binary descriptor).

### 5.5 Matching — `count_good_matches(desc_query, desc_candidate)`

#### BFMatcher with NORM_HAMMING
- Brute-Force Matcher exhaustively compares every query descriptor against every
  candidate descriptor.
- `cv2.NORM_HAMMING` counts differing bits between two binary strings — the
  correct metric for ORB descriptors (float metrics like L2 are meaningless for
  bit arrays).
- `crossCheck=False` is mandatory for `knnMatch`.

#### Lowe's Ratio Test
Introduced by David Lowe in the original SIFT paper (2004) to filter ambiguous
matches:
```
For each query descriptor d_q:
  Find its two nearest neighbours in the candidate set: d_1 (closest), d_2
  Accept match only if:  distance(d_q, d_1) < RATIO_TEST × distance(d_q, d_2)
```
**Intuition**: if the best match is barely better than the second-best, the
descriptor has no discriminative power at this location — discard it. A
ratio of 0.75 is Lowe's recommended default and provides a good
precision/recall trade-off for fingerprint images.

### 5.6 Score Aggregation
- Good-match counts are **summed per subject** across all samples of that
  subject (excluding the query image itself).
- Aggregation over multiple samples makes the decision more robust than relying
  on a single best match.
- `predicted_subject = argmax(subject_scores)`.

### 5.7 Reporting
Printed to standard output:
1. Query image filename and true subject ID.
2. Number of keypoints detected on the query.
3. **Top-10 individual match table**: sorted by good-match count descending.
4. **Subject-level score table**: sorted by aggregate score descending, with
   the predicted subject marked.
5. Final verdict: predicted owner, true owner, CORRECT / INCORRECT flag.
6. Total **elapsed wall-clock time** in seconds.

---

## 6. Configuration Constants

All tunable parameters are declared at the top of the script for easy
adjustment without modifying algorithm logic.

| Constant | Default | Effect if increased | Effect if decreased |
|---|---|---|---|
| `ORB_NFEATURES` | 500 | More keypoints → slower but potentially better recall | Faster but may miss unique minutiae |
| `RATIO_TEST` | 0.75 | Higher → more matches accepted (more recall, less precision) | Lower → fewer, stricter matches |
| `ADAPTIVE_BLOCK` | 11 | Larger neighbourhood → smoother threshold | Smaller → more sensitive to local contrast |
| `ADAPTIVE_C` | 2 | Higher → more pixels become ridge (binary 0) | Lower → fewer ridge pixels |

---

## 7. Patterns Used

| Pattern | Where applied |
|---|---|
| **Pipeline pattern** | Each stage (load → preprocess → extract → match → aggregate → report) is independent and testable |
| **Module-level singleton** | `orb` and `matcher` objects are created once at module load, not per-image call |
| **Guard clause** | `count_good_matches` returns 0 early if descriptors are None or too short |
| **Separation of concerns** | I/O (loading), processing (preprocess/extract), matching, and reporting are separate functions |

---

## 8. Running the Script

```bash
# Must be run from fingerpirnt_detection/ so that relative path resolves
cd fingerpirnt_detection
python fingerprint_recognition.py
```

### Example Output
```
============================================================
FINGERPRINT RECOGNITION SYSTEM
============================================================
Query image : 103_5.tif
True owner  : Subject 103
Comparing against 79 images ...

Query keypoints detected: 487

--- TOP 10 INDIVIDUAL IMAGE MATCHES ---
Image                Subject    Good Matches
--------------------------------------------
103_2.tif            103                  34
103_7.tif            103                  31
103_1.tif            103                  28
...

--- SUBJECT-LEVEL SCORE (aggregated) ---
Subject    Total Score
------------------------
103                168 ◄ PREDICTED
105                 42
...

============================================================
Predicted owner : Subject 103  (score: 168)
True owner      : Subject 103
Result          : ✓ CORRECT
Elapsed time    : 1.2347 seconds
============================================================
```

---

## 9. Limitations & Possible Improvements

| Limitation | Improvement |
|---|---|
| Each image is preprocessed on every run | Cache descriptors in a `.npz` file after first run |
| No image quality filter | Reject images with fewer than N keypoints before matching |
| Single script, no GUI | Add `cv2.imshow` to visualise matches with `cv2.drawMatches` |
| No cross-validation metric | Run over all 80 images as queries and report overall accuracy |
