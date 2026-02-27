import cv2
import os
import random
import time
import numpy as np

# =============================================================================
# CONFIGURATION — adjust these values to tune recognition behaviour
# =============================================================================
IMAGES_DIR      = "fingerprint_images"  # directory with .tif fingerprint images
ORB_NFEATURES   = 500                   # max keypoints extracted by ORB per image
RATIO_TEST      = 0.75                  # Lowe's ratio threshold (lower = stricter)
ADAPTIVE_BLOCK  = 11                    # neighbourhood size for adaptive threshold (must be odd)
ADAPTIVE_C      = 2                     # constant subtracted from weighted mean


# =============================================================================
# STEP 1 — DATASET LOADING
# Parse every .tif file in the images directory and extract:
#   subject_id  → the person owning the fingerprint  (e.g. "101")
#   sample_id   → the specific impression number      (e.g. "3")
# Filename convention: {subject_id}_{sample_id}.tif
# =============================================================================
def load_dataset(images_dir):
    """Return a list of (abs_path, subject_id, sample_id) for every .tif file."""
    entries = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(".tif"):
            continue
        name_parts = fname.replace(".tif", "").split("_")
        subject_id = name_parts[0]
        sample_id  = name_parts[1]
        abs_path   = os.path.join(images_dir, fname)
        entries.append((abs_path, subject_id, sample_id))
    return entries


# =============================================================================
# STEP 2 — PREPROCESSING
# Adaptive thresholding converts the grayscale image to binary by comparing
# each pixel to the Gaussian-weighted mean of its local neighbourhood.
# This enhances the ridge/valley contrast independently of global lighting,
# which is critical for ORB to find stable keypoints on fingerprint patterns.
# =============================================================================
def preprocess(gray_img):
    """Apply adaptive Gaussian thresholding to a grayscale image."""
    # ── ADAPTIVE BINARIZATION ────────────────────────────────────────────────
    # Unlike a global threshold (e.g. all pixels > 128 → white), adaptive
    # thresholding computes a *local* threshold for every pixel based on the
    # intensity values in its neighbourhood window.
    #
    # How it works step by step:
    #   1. For each pixel P at (x, y), take the (ADAPTIVE_BLOCK × ADAPTIVE_BLOCK)
    #      window centred on P.
    #   2. Compute the Gaussian-weighted mean of that window  →  μ_local
    #      (pixels closer to the centre of the window contribute more).
    #   3. Subtract the constant C  →  threshold = μ_local - ADAPTIVE_C
    #   4. If P > threshold  →  output pixel = 255 (white / ridge)
    #      else              →  output pixel = 0   (black / valley)
    #
    # Why this matters for fingerprints:
    #   Fingerprint images often have uneven illumination or sensor noise.
    #   A global threshold would binarize some areas correctly but clip others.
    #   The local neighbourhood adapts to each region, preserving ridge detail
    #   across the entire image regardless of brightness variations.
    return cv2.adaptiveThreshold(
        gray_img,            # source: single-channel 8-bit grayscale image
        255,                 # maxValue: pixel value assigned when condition is met
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # method: Gaussian-weighted neighbourhood mean
        cv2.THRESH_BINARY,  # thresholding type: pixel > threshold → maxValue, else 0
        ADAPTIVE_BLOCK,     # blockSize: side length of the local neighbourhood (must be odd)
        ADAPTIVE_C          # C: constant subtracted from the weighted mean (fine-tunes sensitivity)
    )


# =============================================================================
# STEP 3 — FEATURE EXTRACTION (ORB)
# ORB (Oriented FAST and Rotated BRIEF) detects corners with FAST and computes
# a binary descriptor using BRIEF, adding orientation invariance via the
# intensity centroid method.  It is fast, license-free, and well-suited to
# binary, ridge-structured images like fingerprints.
# =============================================================================
orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)

def extract_features(preprocessed_img):
    """Detect ORB keypoints and compute descriptors on a preprocessed image."""
    keypoints, descriptors = orb.detectAndCompute(preprocessed_img, None)
    return keypoints, descriptors


# =============================================================================
# STEP 4 — DESCRIPTOR MATCHING WITH LOWE'S RATIO TEST
# BFMatcher with NORM_HAMMING is the correct distance metric for binary
# descriptors (ORB produces bit-strings, so Hamming distance counts differing
# bits).  crossCheck=False is mandatory for knnMatch (k=2).
#
# Lowe's Ratio Test (from the SIFT paper): a match between descriptor A and
# its nearest neighbour B is accepted only if the distance to B is significantly
# smaller than the distance to the second-nearest neighbour C, i.e.:
#   distance(A,B) < RATIO_TEST * distance(A,C)
# This discards ambiguous matches and dramatically reduces false positives.
# =============================================================================
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def get_good_matches(desc_query, desc_candidate):
    """Return the list of matches that pass Lowe's ratio test.
    Returning the full list (not just the count) allows callers to use the
    match objects directly with cv2.drawMatches for visualisation.
    """
    # Cannot match if either image produced no descriptors
    if desc_query is None or desc_candidate is None:
        return []
    # knnMatch requires at least k=2 descriptors on both sides
    if len(desc_query) < 2 or len(desc_candidate) < 2:
        return []

    # For every descriptor in the query image, find the 2 closest descriptors
    # in the candidate image (k=2: nearest neighbour m, second-nearest neighbour n).
    # Distance is Hamming (count of differing bits), since ORB descriptors are binary.
    knn_matches = matcher.knnMatch(desc_query, desc_candidate, k=2)

    # ── LOWE'S RATIO TEST ────────────────────────────────────────────────────
    # Proposed by David Lowe in the original SIFT paper (2004).
    # Problem it solves: many descriptor matches are ambiguous — the query
    # descriptor looks almost equally similar to two different candidate
    # descriptors.  Keeping those produces many false positives.
    #
    # The test keeps a match (query → m) only when:
    #   distance(query, m)  <  RATIO_TEST * distance(query, n)
    #
    #   m = nearest neighbour       (smallest Hamming distance)
    #   n = second-nearest neighbour (second smallest)
    #
    # Intuition:
    #   • If m.distance is much smaller than n.distance, the nearest neighbour
    #     is clearly the best option → the match is unambiguous → KEEP.
    #   • If m.distance ≈ n.distance, the descriptor could belong to either
    #     candidate → the match is ambiguous → DISCARD.
    #
    # RATIO_TEST = 0.75 means: accept only if the best match is at least
    # 25% closer than the runner-up.  Lower values are stricter (fewer but
    # more reliable matches); higher values are more permissive.
    return [m for m, n in knn_matches if m.distance < RATIO_TEST * n.distance]
    #        ↑ best match              ↑ ratio test condition


# =============================================================================
# STEP 7 — VISUALISATION
# Builds a three-section canvas:
#   • Header  — colour-coded verdict (green = correct, red = incorrect)
#   • Middle  — cv2.drawMatches side-by-side image with connecting lines
#   • Footer  — elapsed time and score
# The original grayscale images are used (not the preprocessed ones) so the
# actual fingerprint texture is visible behind the match lines.
# =============================================================================
MAX_DRAWN_MATCHES = 40   # cap lines so the image stays readable

def draw_result(query_gray, query_kp, best_gray, best_kp, good_matches,
                query_fname, best_fname, query_subj, predicted_subj,
                best_score, elapsed, correct):
    """Render and display the match visualisation window."""
    HEADER_H = 52
    FOOTER_H = 38

    # ── Side-by-side image with match lines ─────────────────────────────────
    # Limit drawn matches to avoid a cluttered image; sort by distance so the
    # strongest (lowest Hamming distance) matches are drawn first.
    top_matches = sorted(good_matches, key=lambda m: m.distance)[:MAX_DRAWN_MATCHES]

    match_img = cv2.drawMatches(
        query_gray, query_kp,
        best_gray,  best_kp,
        top_matches,
        None,
        matchColor=(0, 230, 0),              # bright green lines
        singlePointColor=(160, 160, 160),    # grey for unmatched keypoints
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    h, w = match_img.shape[:2]

    # ── Assemble canvas ──────────────────────────────────────────────────────
    canvas = np.zeros((HEADER_H + h + FOOTER_H, w, 3), dtype=np.uint8)

    # Place the match image in the middle band
    canvas[HEADER_H: HEADER_H + h, :] = match_img

    # ── Header band ─────────────────────────────────────────────────────────
    # Background: dark green for correct, dark red for incorrect
    hdr_color = (30, 110, 30) if correct else (30, 30, 140)
    cv2.rectangle(canvas, (0, 0), (w, HEADER_H), hdr_color, -1)

    verdict     = "CORRECT" if correct else "INCORRECT"
    verdict_clr = (100, 255, 100) if correct else (100, 130, 255)

    # Left label — query
    cv2.putText(canvas,
                f"QUERY: {query_fname}  (Subject {query_subj})",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    # Right label — best match
    mid = w // 2
    cv2.putText(canvas,
                f"BEST MATCH: {best_fname}  (Subject {predicted_subj})",
                (mid + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    # Verdict centred on second line
    verdict_txt = f"Result: {verdict}"
    txt_sz, _ = cv2.getTextSize(verdict_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(canvas,
                verdict_txt,
                ((w - txt_sz[0]) // 2, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, verdict_clr, 2, cv2.LINE_AA)

    # ── Footer band ─────────────────────────────────────────────────────────
    fy = HEADER_H + h
    cv2.rectangle(canvas, (0, fy), (w, fy + FOOTER_H), (30, 30, 30), -1)
    footer_txt = (f"Good matches (best image): {len(good_matches)}  |  "
                  f"Elapsed: {elapsed:.4f} s  |  Press any key to close")
    cv2.putText(canvas, footer_txt,
                (10, fy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Display ──────────────────────────────────────────────────────────────
    cv2.imshow("Fingerprint Recognition", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# MAIN — RECOGNITION PIPELINE
# =============================================================================
def main():
    # ── Load the full dataset ────────────────────────────────────────────────
    dataset = load_dataset(IMAGES_DIR)
    if len(dataset) < 2:
        print("ERROR: Not enough images in the dataset directory.")
        return

    # ── STEP 2A — Select a random query image ───────────────────────────────
    query_entry              = random.choice(dataset)
    query_path, query_subj, query_sample = query_entry

    print("=" * 60)
    print("FINGERPRINT RECOGNITION SYSTEM")
    print("=" * 60)
    print(f"Query image : {os.path.basename(query_path)}")
    print(f"True owner  : Subject {query_subj}")
    print(f"Comparing against {len(dataset) - 1} images ...\n")

    # ── Extract features from the query image ────────────────────────────────
    query_gray   = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    query_proc   = preprocess(query_gray)          # ← ADAPTIVE BINARIZATION applied here
    query_kp, query_desc = extract_features(query_proc)
    print(f"Query keypoints detected: {len(query_kp)}")

    # ── STEP 5 — Compare query against every other image ────────────────────
    # subject_scores accumulates good-match counts per subject (aggregation
    # across all samples of that subject).
    subject_scores = {}   # {subject_id: int}
    image_results  = []   # [(filename, subject_id, good_count)] — for reporting

    # Track the single best-matching individual image so we can visualise it
    best_individual = {
        "count":   -1,
        "fname":   "",
        "subj":    "",
        "gray":    None,
        "kp":      None,
        "matches": []
    }

    start_time = time.time()

    for path, subj_id, samp_id in dataset:
        # Skip the query image to avoid comparing with itself
        if path == query_path:
            continue

        # Preprocess and extract features from the candidate image
        cand_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cand_proc = preprocess(cand_gray)              # ← ADAPTIVE BINARIZATION applied here
        cand_kp, cand_desc = extract_features(cand_proc)

        # ← RATIO TEST applied inside get_good_matches — only unambiguous matches returned
        good_list = get_good_matches(query_desc, cand_desc)
        good      = len(good_list)

        # Accumulate score for this subject
        subject_scores[subj_id] = subject_scores.get(subj_id, 0) + good

        image_results.append((os.path.basename(path), subj_id, good))

        # Remember this image if it is the best individual match so far
        # (used later for the visual; we store the original gray, not preprocessed)
        if good > best_individual["count"]:
            best_individual.update({
                "count":   good,
                "fname":   os.path.basename(path),
                "subj":    subj_id,
                "gray":    cand_gray,
                "kp":      cand_kp,
                "matches": good_list
            })

    elapsed = time.time() - start_time

    # ── STEP 6 — Identify the owner ──────────────────────────────────────────
    # The subject whose images collectively matched the query the most
    predicted_subj = max(subject_scores, key=subject_scores.get)
    best_score     = subject_scores[predicted_subj]

    # ── STEP 7 — REPORT RESULTS ──────────────────────────────────────────────
    # Sort individual image results by good-match count (descending)
    image_results.sort(key=lambda x: x[2], reverse=True)

    print("\n--- TOP 10 INDIVIDUAL IMAGE MATCHES ---")
    print(f"{'Image':<20} {'Subject':<10} {'Good Matches':>12}")
    print("-" * 44)
    for fname, sid, score in image_results[:10]:
        print(f"{fname:<20} {sid:<10} {score:>12}")

    print("\n--- SUBJECT-LEVEL SCORE (aggregated) ---")
    print(f"{'Subject':<10} {'Total Score':>12}")
    print("-" * 24)
    for sid, total in sorted(subject_scores.items(), key=lambda x: x[1], reverse=True):
        marker = " ◄ PREDICTED" if sid == predicted_subj else ""
        print(f"{sid:<10} {total:>12}{marker}")

    print("\n" + "=" * 60)
    print(f"Predicted owner : Subject {predicted_subj}  (score: {best_score})")
    print(f"True owner      : Subject {query_subj}")
    correct = predicted_subj == query_subj
    print(f"Result          : {'✓ CORRECT' if correct else '✗ INCORRECT'}")
    print(f"Elapsed time    : {elapsed:.4f} seconds")
    print("=" * 60)

    # ── STEP 8 — VISUAL OUTPUT ────────────────────────────────────────────────
    # Show query vs. best individual match with connecting lines and verdict
    print("\nOpening visualisation window (press any key to close)...")
    draw_result(
        query_gray, query_kp,
        best_individual["gray"], best_individual["kp"], best_individual["matches"],
        os.path.basename(query_path), best_individual["fname"],
        query_subj, predicted_subj,
        best_score, elapsed, correct
    )


if __name__ == "__main__":
    main()
