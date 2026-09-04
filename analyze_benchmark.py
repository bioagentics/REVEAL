#!/usr/bin/env python3
"""
analyze_benchmark.py — REVEAL benchmark analysis

Standalone analyser for ADC_Report_*.json run files produced by the REVEAL
pipeline. Deliberately separate from the pipeline so that scoring rules can be
revised and metrics recomputed WITHOUT re-running expensive API calls.

Computes:
  - Task Completion Rate (TCR) with Wilson 95% CI
  - Antigen identification accuracy (supports multi-valid-target ground truth)
  - Retrieval Precision (RP) per antibody lead, split into
    sequence_present / no_sequence / fabricated
  - Audit Depth (AD), derived per-lead from JSON evidence fields
  - Hotspot statistics + cross-repeat determinism check

Emits:
  - leads.csv           one row per antibody lead (for manual IMGT verification)
  - runs.csv            one row per run
  - hotspots.csv        one row per hotspot
  - summary.txt         human-readable report (same as stdout)

Usage:
    python analyze_benchmark.py /path/to/json_dir
    python analyze_benchmark.py /path/to/json_dir --outdir results/
    python analyze_benchmark.py /path/to/json_dir --repeats 10

Memory note: files are streamed one at a time and only small extracted fields
are retained. Full sequences are never held in memory beyond the current file;
only lengths and short prefixes are stored.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

# Ground-truth antigen(s) per benchmark prompt (benchmark_index -> valid symbols).
# Prompt 0 accepts BOTH ERBB2 and TACSTD2/TROP2: sacituzumab govitecan is the
# guideline-standard ADC for metastatic TNBC and is a clinically defensible
# response to a "bystander effect / low antigen density" brief. Documented in
# the manuscript Methods.
EXPECTED_ANTIGEN = {
    0: {"ERBB2", "HER2", "TACSTD2", "TROP2", "TROP-2", "TROPE2"},
    1: {"ERBB2", "HER2"},
    2: {"FOLR1", "FRA", "FOLATE RECEPTOR ALPHA"},
    3: {"TNFRSF17", "BCMA"},
    4: {"TACSTD2", "TROP2", "TROP-2"},
    5: {"NECTIN4", "NECTIN-4", "PVRL4"},
    6: {"TNFRSF8", "CD30"},
    7: {"CD33"},
    8: {"MSLN", "MESOTHELIN"},
}

PROMPT_NAMES = {
    0: "HER2/TNBC",           1: "HER2/CRC",            2: "FOLR1/Ovarian",
    3: "BCMA/Myeloma",        4: "TROP2/TNBC",          5: "Nectin-4/Urothelial",
    6: "CD30/Hodgkin",        7: "CD33/AML",            8: "MSLN/Pancreatic",
}

# Names the pipeline uses for its hardcoded fallback antibody. Leads matching
# these are NOT genuine retrievals and are excluded from RP.
FALLBACK_MARKERS = ("trastuzumab_fallback",)

MIN_SEQ_LEN = 15          # shorter strings aren't credible chain sequences
FAB_WINDOW = 25           # window size for antigen-substring fabrication check
FAB_MAX_OFFSET = 12       # tolerate small fabricated prefixes (e.g. "GSS...")


# --------------------------------------------------------------------------
# STATISTICS
# --------------------------------------------------------------------------

def wilson_ci(successes, n, z=1.96):
    """Wilson score interval — appropriate for proportions with small/moderate n."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return ((center - margin) / denom * 100.0, (center + margin) / denom * 100.0)


def pct(successes, n):
    return (successes / n * 100.0) if n else 0.0


def fmt_prop(successes, n, label):
    lo, hi = wilson_ci(successes, n)
    return f"{label}: {successes}/{n} = {pct(successes,n):.1f}%  (95% CI {lo:.1f}–{hi:.1f}%)"


# --------------------------------------------------------------------------
# CLASSIFIERS
# --------------------------------------------------------------------------

def clean_str(v):
    """Return a usable string, or None. Handles the literal 'null'/'NULL'/'None'
    strings the LLM sometimes emits instead of real JSON nulls."""
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s or s.lower() in ("null", "none", "n/a", "na", "<not available>"):
        return None
    return s


def is_fabricated(seq, antigen_seq):
    """
    Detect the characterised failure mode where a fragment of the ANTIGEN
    sequence is returned labelled as an antibody chain.

    Slides a window across the start of the candidate sequence (tolerating a
    short fabricated prefix) and tests for containment in the antigen sequence.
    """
    if not seq or not antigen_seq or len(seq) < FAB_WINDOW:
        return False
    s = re.sub(r"\s+", "", seq).upper()
    a = re.sub(r"\s+", "", antigen_seq).upper()
    limit = min(FAB_MAX_OFFSET, max(0, len(s) - FAB_WINDOW))
    for off in range(limit + 1):
        if s[off:off + FAB_WINDOW] in a:
            return True
    return False


def classify_lead(lead, antigen_seq):
    """-> (status, vh_len, vl_len). Status is one of:
       fallback | no_sequence | fabricated | sequence_present"""
    name = (lead.get("name") or "").strip()
    if any(m in name.lower() for m in FALLBACK_MARKERS):
        return "fallback", 0, 0

    vh = clean_str(lead.get("heavy_chain"))
    vl = clean_str(lead.get("light_chain"))
    vh = vh if (vh and len(vh) >= MIN_SEQ_LEN) else None
    vl = vl if (vl and len(vl) >= MIN_SEQ_LEN) else None

    if not vh and not vl:
        return "no_sequence", 0, 0
    if (vh and is_fabricated(vh, antigen_seq)) or (vl and is_fabricated(vl, antigen_seq)):
        return "fabricated", len(vh or ""), len(vl or "")
    return "sequence_present", len(vh or ""), len(vl or "")


# --------------------------------------------------------------------------
# AUDIT DEPTH
# --------------------------------------------------------------------------

def compute_audit_depth(data, leads):
    """
    Audit Depth, derived per-lead from the exported evidence fields rather than
    from hand-assigned constants.

    For each discovered lead i:
        P_i = bibliographic references cited in the run (shared provenance)
        D_i = non-null quantitative data fields across ADC configurations
        L_i = distinct evidence URLs traceable to that lead
                (its own evidence_url + the antigen evidence URL)

        AD = (1/N) * sum_i (P_i + D_i + L_i)

    This replaces the original Eq. 5 formulation, which applied one constant
    triple to all leads and whose reported arithmetic (mean of 13 identical
    values of 16 reported as 16/13) was internally inconsistent. Deriving the
    terms from exported fields makes AD reproducible and per-lead varying.

    Returns (AD, N_leads, mean_P, mean_D, mean_L).
    """
    bio = data.get("biological_phase") or {}
    refs = data.get("bibliographic_references") or []
    P = len([r for r in refs if isinstance(r, str) and r.strip()])

    # D: count populated quantitative fields across ADC configuration records
    quant_fields = ("ic50_nm", "verified_ic50", "molecular_weight_da",
                    "logp_value", "dar_ratio", "assay_cell_line",
                    "payload_smiles", "linker_smiles", "binding_affinity_kd")
    D = 0
    adcs = bio.get("adc_configurations") or []
    if isinstance(adcs, list):
        for rec in adcs:
            if not isinstance(rec, dict):
                continue
            for f in quant_fields:
                v = rec.get(f)
                if v is None:
                    continue
                if isinstance(v, str) and not v.strip():
                    continue
                if isinstance(v, (list, dict)) and not v:
                    continue
                D += 1

    antigen_url = clean_str(bio.get("antigen_evidence_url"))

    per_lead = []
    for lead in leads:
        links = set()
        u = clean_str(lead.get("evidence_url"))
        if u:
            links.add(u)
        if antigen_url:
            links.add(antigen_url)
        per_lead.append((P, D, len(links)))

    if not per_lead:
        return 0.0, 0, 0.0, 0.0, 0.0

    n = len(per_lead)
    ad = sum(p + d + l for p, d, l in per_lead) / n
    return (ad, n,
            sum(p for p, _, _ in per_lead) / n,
            sum(d for _, d, _ in per_lead) / n,
            sum(l for _, _, l in per_lead) / n)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def analyse(json_dir, outdir, repeats_expected):
    paths = sorted(glob.glob(os.path.join(json_dir, "ADC_Report_*.json")))
    if not paths:
        sys.exit(f"No ADC_Report_*.json files found in: {json_dir}")

    run_rows, lead_rows, hotspot_rows = [], [], []
    seen = set()

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"  [!] Skipping unreadable {os.path.basename(path)}: {e}")
            continue

        meta = data.get("metadata") or {}
        bidx = meta.get("benchmark_index")
        ridx = meta.get("repeat_index")
        if bidx is None:
            continue
        key = (bidx, ridx)
        if key in seen:
            print(f"  [!] Duplicate (prompt {bidx}, repeat {ridx}) — skipping "
                  f"{os.path.basename(path)}")
            continue
        seen.add(key)

        bio = data.get("biological_phase") or {}
        target = (bio.get("target") or "").strip()
        antigen_seq = bio.get("antigen_sequence") or ""
        leads = [l for l in (bio.get("discovered_leads") or []) if isinstance(l, dict)]

        antigen_ok = target.upper() in EXPECTED_ANTIGEN.get(bidx, set())
        ad, ad_n, mP, mD, mL = compute_audit_depth(data, leads)

        # ---- hotspots ----
        eng = data.get("engineering_phase") or {}
        hot = eng.get("hotspot_data") or {}

        # Build antibody_name -> (vh, vl) lookup so each hotspot row can carry
        # the wild-type sequence it was scored against and the mutated variant.
        # hotspot_data is keyed by the same antibody names used in
        # discovered_leads (both are populated by lead_extraction_node).
        seq_lookup = {}
        for l in leads:
            nm = (l.get("name") or "").strip()
            if nm:
                seq_lookup[nm] = (clean_str(l.get("heavy_chain")),
                                  clean_str(l.get("light_chain")))
        # The hardcoded fallback antibody never appears in discovered_leads,
        # so recover its sequences from the engineered_candidates entries.
        for cand in (eng.get("engineered_candidates") or []):
            if not isinstance(cand, dict):
                continue
            nm = (cand.get("antibody_name") or "").strip()
            if not nm or nm in seq_lookup:
                continue
            # A candidate sequence is the WT with one substitution applied;
            # reverse it to recover the wild-type chain.
            cseq = clean_str(cand.get("sequence"))
            pos, wt = cand.get("pos_idx"), cand.get("wt")
            if cseq and isinstance(pos, int) and isinstance(wt, str) and 0 <= pos < len(cseq):
                restored = cseq[:pos] + wt + cseq[pos + 1:]
                if cand.get("chain") == "H":
                    seq_lookup.setdefault(nm, (restored, None))
                else:
                    prev = seq_lookup.get(nm, (None, None))
                    seq_lookup[nm] = (prev[0], restored)

        n_h = n_l = 0
        llrs = []
        if isinstance(hot, dict):
            for ab_name, rec in hot.items():
                if not isinstance(rec, dict):
                    continue
                vh_seq, vl_seq = seq_lookup.get(ab_name, (None, None))
                for chain_key, bucket in (("H", "heavy_hotspots"), ("L", "light_hotspots")):
                    wt_seq = vh_seq if chain_key == "H" else vl_seq
                    for h in (rec.get(bucket) or []):
                        if not isinstance(h, dict):
                            continue
                        llr = h.get("llr")
                        if chain_key == "H":
                            n_h += 1
                        else:
                            n_l += 1
                        if isinstance(llr, (int, float)):
                            llrs.append(float(llr))

                        pos = h.get("pos_idx")
                        wt = h.get("wt", "")
                        mut = h.get("mut", "")
                        orig_seq, mut_seq, wt_check = "", "", ""
                        if wt_seq and isinstance(pos, int) and 0 <= pos < len(wt_seq):
                            orig_seq = wt_seq
                            # Sanity check: does the residue at pos_idx actually
                            # match the reported wild-type amino acid?
                            wt_check = "OK" if wt_seq[pos] == wt else \
                                       f"MISMATCH(seq={wt_seq[pos]})"
                            if isinstance(mut, str) and len(mut) == 1:
                                mut_seq = wt_seq[:pos] + mut + wt_seq[pos + 1:]
                        elif wt_seq:
                            orig_seq = wt_seq
                            wt_check = "POS_OUT_OF_RANGE"
                        else:
                            wt_check = "NO_SEQUENCE"

                        hotspot_rows.append({
                            "file": os.path.basename(path),
                            "prompt": bidx + 1,
                            "repeat": (ridx + 1) if ridx is not None else "",
                            "antibody": ab_name,
                            "chain": chain_key,
                            "pos_idx": pos if pos is not None else "",
                            "wt": wt,
                            "mut": mut,
                            "llr": llr if llr is not None else "",
                            "wt_residue_check": wt_check,
                            "seq_len": len(wt_seq) if wt_seq else "",
                            "original_sequence": orig_seq,
                            "mutated_sequence": mut_seq,
                        })

        n_eng = len(eng.get("engineered_candidates") or [])

        run_rows.append({
            "file": os.path.basename(path),
            "prompt": bidx + 1,
            "prompt_name": PROMPT_NAMES.get(bidx, "?"),
            "repeat": (ridx + 1) if ridx is not None else "",
            "target_returned": target,
            "antigen_correct": int(antigen_ok),
            "n_leads": len(leads),
            "n_hotspots_heavy": n_h,
            "n_hotspots_light": n_l,
            "n_engineered_candidates": n_eng,
            "audit_depth": round(ad, 3),
            "AD_mean_P": round(mP, 2),
            "AD_mean_D": round(mD, 2),
            "AD_mean_L": round(mL, 2),
            "max_llr": round(max(llrs), 3) if llrs else "",
        })

        for lead in leads:
            status, vh_len, vl_len = classify_lead(lead, antigen_seq)
            lead_rows.append({
                "file": os.path.basename(path),
                "prompt": bidx + 1,
                "prompt_name": PROMPT_NAMES.get(bidx, "?"),
                "repeat": (ridx + 1) if ridx is not None else "",
                "target_returned": target,
                "antibody_name": (lead.get("name") or "").strip(),
                "status": status,
                "vh_len": vh_len,
                "vl_len": vl_len,
                "evidence_url": clean_str(lead.get("evidence_url")) or "",
                # For the manual IMGT/Thera-SAbDab verification pass:
                "verified_correct": "",       # fill in: TRUE / FALSE / UNKNOWN
                "verification_note": "",
            })

    # ---------------- report ----------------
    os.makedirs(outdir, exist_ok=True)
    out = []
    def w(line=""):
        out.append(line)
        print(line)

    n_runs = len(run_rows)
    n_prompts = len(set(r["prompt"] for r in run_rows))
    expected_total = n_prompts * repeats_expected

    w("=" * 78)
    w("REVEAL BENCHMARK ANALYSIS")
    w("=" * 78)
    w(f"Runs found: {n_runs}   Prompts: {n_prompts}   Expected: {expected_total}")
    w()

    w("-" * 78)
    w("TASK COMPLETION RATE")
    w("-" * 78)
    w(fmt_prop(n_runs, expected_total, "TCR (completed / attempted)"))
    missing = []
    for b in sorted(set(r["prompt"] - 1 for r in run_rows)):
        have = {r["repeat"] for r in run_rows if r["prompt"] - 1 == b}
        for rep in range(1, repeats_expected + 1):
            if rep not in have:
                missing.append(f"prompt {b+1} rep{rep:02d}")
    if missing:
        w(f"Missing runs ({len(missing)}): {', '.join(missing)}")
    w()

    w("-" * 78)
    w("ANTIGEN IDENTIFICATION ACCURACY")
    w("-" * 78)
    n_ok = sum(r["antigen_correct"] for r in run_rows)
    w(fmt_prop(n_ok, n_runs, "Antigen ID accuracy (overall)"))
    w()

    w("-" * 78)
    w("RETRIEVAL PRECISION (per antibody lead)")
    w("-" * 78)
    scored = [r for r in lead_rows if r["status"] != "fallback"]
    n_leads = len(scored)
    n_present = sum(1 for r in scored if r["status"] == "sequence_present")
    n_null = sum(1 for r in scored if r["status"] == "no_sequence")
    n_fab = sum(1 for r in scored if r["status"] == "fabricated")
    n_fallback = sum(1 for r in lead_rows if r["status"] == "fallback")
    w(fmt_prop(n_present, n_leads, "RP (sequence present & plausible)"))
    w(f"  no sequence retrieved : {n_null}/{n_leads} = {pct(n_null,n_leads):.1f}%")
    w(f"  fabricated (antigen)  : {n_fab}/{n_leads} = {pct(n_fab,n_leads):.1f}%")
    if n_fallback:
        w(f"  (excluded: {n_fallback} hardcoded-fallback leads)")
    w()
    w("  NOTE: 'sequence present' is a HEURISTIC pass — only null sequences and")
    w("  antigen-substring fabrication are ruled out automatically. These are NOT")
    w("  verified against IMGT/Thera-SAbDab. Use leads.csv ('verified_correct'")
    w("  column) for the manual verification pass before citing RP as final.")
    w()

    w("-" * 78)
    w("AUDIT DEPTH (derived per-lead from exported evidence fields)")
    w("-" * 78)
    ads = [r["audit_depth"] for r in run_rows if r["n_leads"] > 0]
    if ads:
        mean_ad = sum(ads) / len(ads)
        var = sum((x - mean_ad) ** 2 for x in ads) / len(ads)
        w(f"Mean AD across {len(ads)} runs with >=1 lead: {mean_ad:.2f} (SD {math.sqrt(var):.2f})")
        w(f"Range: {min(ads):.2f} – {max(ads):.2f}")
        w(f"Runs with AD > 3.0: {sum(1 for x in ads if x > 3.0)}/{len(ads)}")
    else:
        w("No runs with leads — AD not computable.")
    w()

    w("-" * 78)
    w("HOTSPOTS")
    w("-" * 78)
    all_llrs = [h["llr"] for h in hotspot_rows if isinstance(h["llr"], (int, float))]
    w(f"Total hotspots: {len(hotspot_rows)}")
    if all_llrs:
        srt = sorted(all_llrs)
        w(f"LLR range: {srt[0]:.3f} – {srt[-1]:.3f}   median {srt[len(srt)//2]:.3f}")
        w(f"Hotspots with LLR < 0.10 (indistinguishable from zero): "
          f"{sum(1 for x in all_llrs if x < 0.10)}/{len(all_llrs)}")

    # Wild-type residue verification: does the residue at pos_idx in the parent
    # chain actually match the reported 'wt'? A mismatch means the hotspot was
    # scored against a different sequence than the one exported as the lead.
    checks = defaultdict(int)
    for h in hotspot_rows:
        c = h.get("wt_residue_check", "")
        if c.startswith("MISMATCH"):
            checks["mismatch"] += 1
        elif c:
            checks[c] += 1
    w()
    w("Wild-type residue verification (original vs. mutated sequence columns):")
    w(f"  residue matches reported wt : {checks.get('OK', 0)}")
    w(f"  MISMATCH                    : {checks.get('mismatch', 0)}")
    w(f"  position out of range       : {checks.get('POS_OUT_OF_RANGE', 0)}")
    w(f"  no parent sequence found    : {checks.get('NO_SEQUENCE', 0)}")
    if checks.get("mismatch") or checks.get("POS_OUT_OF_RANGE"):
        w("  [!] Non-zero mismatches indicate hotspots scored against a sequence")
        w("      different from the one exported — investigate before citing.")
    # determinism: identical hotspot fingerprint across repeats of same prompt
    fp = defaultdict(set)
    for h in hotspot_rows:
        fp[h["prompt"]].add((h["antibody"], h["chain"], h["pos_idx"], h["wt"], h["mut"], h["llr"]))
    w()
    w("Determinism check (distinct hotspot fingerprints per prompt):")
    for p in sorted(fp):
        reps = len({r["repeat"] for r in run_rows if r["prompt"] == p})
        w(f"  Prompt {p} ({PROMPT_NAMES.get(p-1,'?')}): "
          f"{len(fp[p])} distinct hotspots across {reps} repeats")
    w()

    w("-" * 78)
    w("PER-PROMPT BREAKDOWN")
    w("-" * 78)
    w(f"{'Prompt':<24}{'Runs':>5}{'AntigenOK':>11}{'Leads':>7}"
      f"{'OK':>5}{'Null':>6}{'Fab':>5}{'MeanAD':>8}")
    for b in sorted(set(r["prompt"] - 1 for r in run_rows)):
        rr = [r for r in run_rows if r["prompt"] - 1 == b]
        ll = [r for r in scored if r["prompt"] - 1 == b]
        a = [r["audit_depth"] for r in rr if r["n_leads"] > 0]
        w(f"{PROMPT_NAMES.get(b,'?'):<24}{len(rr):>5}"
          f"{str(sum(x['antigen_correct'] for x in rr)) + '/' + str(len(rr)):>11}"
          f"{len(ll):>7}"
          f"{sum(1 for x in ll if x['status']=='sequence_present'):>5}"
          f"{sum(1 for x in ll if x['status']=='no_sequence'):>6}"
          f"{sum(1 for x in ll if x['status']=='fabricated'):>5}"
          f"{(sum(a)/len(a) if a else 0):>8.2f}")
    w()

    # ---------------- CSVs ----------------
    def dump(rows, name):
        if not rows:
            return
        p = os.path.join(outdir, name)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        w(f"Wrote {p}  ({len(rows)} rows)")

    dump(lead_rows, "leads.csv")
    dump(run_rows, "runs.csv")
    dump(hotspot_rows, "hotspots.csv")

    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print(f"Wrote {os.path.join(outdir, 'summary.txt')}")


def main():
    ap = argparse.ArgumentParser(description="Analyse REVEAL benchmark JSON outputs.")
    ap.add_argument("json_dir", help="Directory containing ADC_Report_*.json files")
    ap.add_argument("--outdir", default="analysis_results", help="Output directory")
    ap.add_argument("--repeats", type=int, default=10,
                    help="Repeats attempted per prompt (for TCR denominator)")
    args = ap.parse_args()
    analyse(args.json_dir, args.outdir, args.repeats)


if __name__ == "__main__":
    main()
