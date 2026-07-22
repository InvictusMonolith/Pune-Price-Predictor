# Dev Log — Coordinate Lookup & Location Grouping Experiment

## What I Set Out to Do
Two remaining items from the Phase 2 roadmap:
1. Test whether expanding location grouping from top-20 to top-40 localities improves model accuracy.
2. Replace manual lat/long entry in the Streamlit app with an automatic per-locality coordinate lookup.

---

## Experiment 1: Expanding Location Grouping (Top-20 → Top-40)

**Hypothesis:** More granular location categories would let the model capture more location-specific price variation.

**What happened:** R² spiked from 0.4088 to 0.6910 — a jump too large to be a real improvement given that location dummies were already known to be minor features (each under 2.3% importance in the original analysis).

**Root cause found:** Data leakage. A `price_cat` column (a binned/categorized version of the target `price`) was still present in the dataframe when features were selected. Only `price` itself was dropped before training — `price_cat`, derived directly from the target, was left in, making the "improvement" fake.

**Fix:** Dropped `price_cat` alongside `price` before rebuilding the feature set.

**Real result after fixing the leak:** R² actually landed at 0.3987 — *worse* than the original top-20 grouping (0.4088). More location categories diluted the training data per locality without adding real signal.

**Decision:** Reverted to top-20 location grouping. Documenting this as a legitimate negative result — the hypothesis was reasonable but the data didn't support it.

**Lesson:** An unexpectedly large accuracy jump is a signal to investigate for leakage, not a reason to celebrate. Always double-check that every column derived from or correlated with the target is excluded from features — not just the target column itself.

---

## Experiment 2: Per-Locality Coordinate Lookup

**Problem:** The Streamlit app previously required manually typing latitude/longitude, defaulting to one generic Pune coordinate regardless of the selected locality. Since lat/long together drive ~24% of the model's predictions, this meaningfully distorted results for any locality other than the default.

**Approach:**
- Loaded the raw dataset independently of the app's working dataframe (which no longer has a `location` column post-encoding).
- Filtered to Pune's valid coordinate bounding box to exclude the known geographic outlier found earlier in the project.
- Built a lookup table: `raw.groupby("location")[["latitude", "longitude"]].mean()`.

**Bug hit:** After adding the lookup table, coordinates in the app still weren't changing when a different locality was selected.

**Root cause:** The lookup table (`locality_coords`) was computed but never actually connected to anything — the `latitude`/`longitude` inputs were still hardcoded via `st.number_input(..., value=18.5204)`, completely disconnected from the new lookup logic. Classic case of writing a fix without wiring it into the code path that actually runs.

**Fix:** Replaced the hardcoded default values with a lookup keyed on the selected locality, with a fallback to the dataset-wide average coordinate for localities not in the top-N list (e.g., "Other").

**Secondary improvement:** Wrapped the CSV load + groupby in `@st.cache_data`, since Streamlit reruns the entire script on every widget interaction — without caching, the full dataset was being reloaded and regrouped on every single click.

**Lesson:** Writing a fix and *using* the fix are two different steps. A variable existing in the code doesn't mean anything downstream reads it — always trace whether new logic is actually wired into the execution path, not just present in the file.

---

## Summary of Mistakes (and why they're worth keeping in the log)
1. Assumed a big accuracy jump was good news before checking for data leakage.
2. Added a lookup table but forgot to connect it to the UI inputs it was meant to replace.

Both were caught by testing the actual behavior rather than trusting that the code *looked* right — the same debugging discipline that's shown up throughout this project.
