"""
nodes_reporting.py

Final reporting phase: the human-readable grand summary printout and the
JSON archival export (ADC_Report_*.json), plus a per-run text summary
(ADC_RunSummary_*.txt) written alongside each JSON.

NOTE: the CSV export that used to run alongside the JSON export
(ADC_EngineeredSequences_*.csv, plus its _find_adc_for_antibody helper)
has been intentionally removed per request -- everything else in
json_export_node is unchanged.

The per-run summary below is INTENTIONALLY SELF-CONTAINED: it does not
import anything from benchmark_analysis.py (that file stays a fully
independent, standalone post-hoc analyser -- run manually, never called
from pipeline code). The lead-classification and Audit Depth logic here is
a deliberate, small duplicate of the same-named functions in
benchmark_analysis.py, kept in sync by hand rather than by import.

Split out of adcAgent_REVEAL_v2.py without changing any other logic.
"""

import re
from state import MasterState
from config import llm

# --------------------------------------------------------------------------
# Per-run summary scoring helpers (self-contained duplicate of the relevant
# pieces of benchmark_analysis.py -- see note above for why).
# --------------------------------------------------------------------------

_FALLBACK_MARKERS = ("trastuzumab_fallback",)
_MIN_SEQ_LEN = 15
_FAB_WINDOW = 25
_FAB_MAX_OFFSET = 12


def _clean_str(v):
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s or s.lower() in ("null", "none", "n/a", "na", "<not available>"):
        return None
    return s


def _is_fabricated(seq, antigen_seq):
    if not seq or not antigen_seq or len(seq) < _FAB_WINDOW:
        return False
    s = re.sub(r"\s+", "", seq).upper()
    a = re.sub(r"\s+", "", antigen_seq).upper()
    limit = min(_FAB_MAX_OFFSET, max(0, len(s) - _FAB_WINDOW))
    for off in range(limit + 1):
        if s[off:off + _FAB_WINDOW] in a:
            return True
    return False


def _classify_lead(lead, antigen_seq):
    """-> (status, vh_len, vl_len). Status one of:
    fallback | no_sequence | fabricated | sequence_present"""
    name = (lead.get("name") or "").strip()
    if any(m in name.lower() for m in _FALLBACK_MARKERS):
        return "fallback", 0, 0
    vh = _clean_str(lead.get("heavy_chain"))
    vl = _clean_str(lead.get("light_chain"))
    vh = vh if (vh and len(vh) >= _MIN_SEQ_LEN) else None
    vl = vl if (vl and len(vl) >= _MIN_SEQ_LEN) else None
    if not vh and not vl:
        return "no_sequence", 0, 0
    if (vh and _is_fabricated(vh, antigen_seq)) or (vl and _is_fabricated(vl, antigen_seq)):
        return "fabricated", len(vh or ""), len(vl or "")
    return "sequence_present", len(vh or ""), len(vl or "")


def _compute_audit_depth(data, leads):
    """Same formula as benchmark_analysis.compute_audit_depth():
    AD = (1/N) * sum_i (P_i + D_i + L_i). Returns (AD, N, mean_P, mean_D, mean_L)."""
    bio = data.get("biological_phase") or {}
    refs = data.get("bibliographic_references") or []
    P = len([r for r in refs if isinstance(r, str) and r.strip()])

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

    antigen_url = _clean_str(bio.get("antigen_evidence_url"))
    per_lead = []
    for lead in leads:
        links = set()
        u = _clean_str(lead.get("evidence_url"))
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


def _render_run_summary(export_data, filename_label=""):
    """Single-run text report -- no Wilson CI / per-prompt table / cross-repeat
    determinism, since those need more than one run's worth of data. If you
    want the full aggregate report across many runs, use
    benchmark_analysis.py directly (standalone, unchanged, not imported here)."""
    meta = export_data.get("metadata") or {}
    bidx = meta.get("benchmark_index")
    ridx = meta.get("repeat_index")
    bio = export_data.get("biological_phase") or {}
    eng = export_data.get("engineering_phase") or {}

    target = (bio.get("target") or "").strip()
    antigen_seq = bio.get("antigen_sequence") or ""
    leads = [l for l in (bio.get("discovered_leads") or []) if isinstance(l, dict)]

    ad, ad_n, mP, mD, mL = _compute_audit_depth(export_data, leads)

    hot = eng.get("hotspot_data") or {}
    llrs, n_h, n_l = [], 0, 0
    if isinstance(hot, dict):
        for ab_name, rec in hot.items():
            if not isinstance(rec, dict):
                continue
            for chain_key, bucket in (("H", "heavy_hotspots"), ("L", "light_hotspots")):
                for h in (rec.get(bucket) or []):
                    if not isinstance(h, dict):
                        continue
                    if chain_key == "H":
                        n_h += 1
                    else:
                        n_l += 1
                    llr = h.get("llr")
                    if isinstance(llr, (int, float)):
                        llrs.append(float(llr))

    n_eng = len(eng.get("engineered_candidates") or [])

    out = []
    def w(line=""):
        out.append(line)

    w("=" * 78)
    w("REVEAL RUN SUMMARY")
    if filename_label:
        w(f"Paired JSON: {filename_label}")
    w("=" * 78)
    w(f"Timestamp: {meta.get('timestamp', '?')}")
    w(f"Prompt index: {(bidx + 1) if bidx is not None else '-'}   "
      f"Repeat: {(ridx + 1) if ridx is not None else '-'}")
    w(f"Iterations: clinical={meta.get('iterations', 0)}")
    w()
    w(f"Target antigen returned: {target or '(none)'}")
    w()

    w("-" * 78)
    w(f"ANTIBODY LEADS ({len(leads)})")
    w("-" * 78)
    n_present = n_null = n_fab = n_fallback = 0
    for lead in leads:
        status, vh_len, vl_len = _classify_lead(lead, antigen_seq)
        name = (lead.get("name") or "").strip() or "(unnamed)"
        w(f"  {name:<28} status={status:<16} VH={vh_len:>4}aa  VL={vl_len:>4}aa")
        if status == "sequence_present":
            n_present += 1
        elif status == "no_sequence":
            n_null += 1
        elif status == "fabricated":
            n_fab += 1
        elif status == "fallback":
            n_fallback += 1
    w()
    w(f"  present={n_present}  no_sequence={n_null}  fabricated={n_fab}  fallback={n_fallback}")

    rejected = bio.get("rejected_leads") or []
    if rejected:
        w()
        w(f"  Rejected by validation gate ({len(rejected)}):")
        for r in rejected:
            w(f"    {r}")
    w()

    w("-" * 78)
    w("AUDIT DEPTH")
    w("-" * 78)
    w(f"AD = {ad:.2f}  (N={ad_n} leads; mean P={mP:.2f}, mean D={mD:.2f}, mean L={mL:.2f})")
    w()

    w("-" * 78)
    w("HOTSPOTS & ENGINEERING")
    w("-" * 78)
    w(f"Heavy-chain hotspots: {n_h}   Light-chain hotspots: {n_l}   "
      f"Engineered candidates: {n_eng}")
    if llrs:
        srt = sorted(llrs)
        w(f"LLR range: {srt[0]:.3f} – {srt[-1]:.3f}   median {srt[len(srt) // 2]:.3f}")
        w(f"LLR < 0.10 (indistinguishable from zero): "
          f"{sum(1 for x in llrs if x < 0.10)}/{len(llrs)}")
    w()

    refs = export_data.get("bibliographic_references") or []
    w("-" * 78)
    w(f"References cited: {len(refs)}")
    w("-" * 78)

    return "\n".join(out)


def grand_summary_presenter_node(state: MasterState):
    print("\n" + "═"*80)
    print(" " * 20 + "PHD RESEARCH: AGENTIC-ADC DESIGN REPORT")
    print("═"*80)

    # 1. TARGET ANTIGEN PROFILE
    antigen = state.get("target_antigen", "Unknown")
    ant_seq = state.get("antigen_sequence", "No sequence retrieved")
    ant_url = state.get("antigen_evidence_url", "N/A")

    print(f"\n[SECTION 1: TARGET CHARACTERIZATION]")
    print(f"• Target Antigen:  {antigen}")
    print(f"• Source/Evidence: {ant_url}")
    print(f"• Sequence Snippet ({len(ant_seq) if ant_seq else 0} AA):")
    if ant_seq and len(ant_seq) > 20:
        print(f"  {ant_seq[:70]}...")
    else:
        print(f"  {ant_seq}")

    # 2. ITERATIVE REASONING (Clinical & Biologist Pass)
    print(f"\n[SECTION 2: REASONING & REVISION LOG]")
    # Grabbing the first HumanMessage and the last few AI responses to show growth
    msgs = state.get("messages", [])
    if msgs:
        print(f"• Initial Requirement: {msgs[0].content[:100]}...")
        # Summarizing the revision loop
        print(f"• Iteration Counts:   [Clinical: {state.get('iteration_count', 0)}] [Biologist: {state.get('biologist_iterations', 0)}]")

    # 3. ANTIBODY LEAD REPOSITORY
    leads = state.get("antibody_leads", [])
    h_map = state.get("heavy_chains", {})
    l_map = state.get("light_chains", {})
    u_map = state.get("ab_sequence_evidence_urls", {})  # FIX: was reading non-existent "evidence_urls"

    print(f"\n[SECTION 3: DISCOVERED ANTIBODY LEADS ({len(leads)})]")
    for i, name in enumerate(leads, 1):
        print(f"\n  {i}. {name.upper()}")
        vh = h_map.get(name, "Missing")
        vl = l_map.get(name, "Missing")
        print(f"     VH: {vh[:50]}..." if len(vh) > 50 else f"     VH: {vh}")
        print(f"     VL: {vl[:50]}..." if len(vl) > 50 else f"     VL: {vl}")
        print(f"     Source: {u_map.get(name, 'Direct Extraction')}")

    # 4. ADC BENCHMARKS & BIOACTIVITY
    adcs = state.get("found_adcs", [])
    print(f"\n[SECTION 4: CLINICAL ADC BENCHMARKS & PHARMACOLOGY]")

    # The 'or "N/A"' handles None; the str() ensures the :< padding works

    if isinstance(adcs, list) and adcs:
        # Define table header
        print(f"{'ADC Name':<25} | {'Payload':<15} | {'IC50 (nM)':<10} | {'LogP':<6}")
        print("-" * 70)
        for adc in adcs:
            name = str(adc.get("adc_name") or "N/A")[:24]
            p_name = str(adc.get("payload_name") or "N/A")[:14]
            ic50 = str(adc.get("ic50_nm") or "N/A")
            logp = str(adc.get("logp_value") or "N/A")
            print(f"{name:<25} | {p_name:<15} | {ic50:<10} | {logp:<6}")

            # Print SMILES if available
            if adc.get("payload_smiles"):
                print(f"     └ Payload SMILES: {adc.get('payload_smiles')[:60]}...")
    else:
        print("   No structured ADC clinical data found.")

    # SECTION 4: NOVELTY & GAP ANALYSIS (The PhD Highlight)
    print(f"\n[SECTION 5] SYSTEMATIC GAP ANALYSIS")
    # We use the LLM to generate a final 'Insight' based on the verified data
    insight_prompt = f"Analyze these verified ADCs: {adcs}. Identify one research gap for this target."
    gap_insight = llm.invoke(insight_prompt).content
    print(f"    • Insight: {gap_insight}")



    #6. Identified Mutations and updated sequence
    print(f"\n[SECTION 6: IN SILICO SEQUENCE ENGINEERING]")
    # NOTE: hotspot_reports is now keyed by antibody name (see
    # antibody_optimization_node); optimized_sequences (built from it in
    # mutation_application_node) already carries an "antibody_name" field
    # per lead, so this table naturally covers every antibody scanned, not
    # just one.

    # Header for the Mutagenesis Table
    print(f"{'Antibody':<15} | {'Chain':<8} | {'Residue':<8} | {'WT':<4} -> {'MUT':<4} | {'LLR Score':<12} | {'Rationale'}")
    print("-" * 90)

    # We display the applied mutations that were selected for the leads
    optimized = state.get('optimized_sequences', [])
    for lead in optimized:
        chain = "Heavy" if lead.get("chain") == "H" else "Light"
        antibody_name = lead.get("antibody_name", "Unknown")
        print(f"{antibody_name:<15} | {chain:<8} | {lead['pos_idx']:<8} | {lead['wt']:<4} -> {lead['mut']:<4} | "
              f"{lead['llr_score']:<12} | Predicted Fitness Gain")

    # 3. Final Result: Updated Engineered Sequences
    print(f"\n[SECTION 7: FINAL ENGINEERED SEQUENCES]")
    print("-" * 80)
    for lead in optimized:
        print(f"\n>> LEAD ID: {lead['id']} (Antibody: {lead.get('antibody_name', 'Unknown')}, Score: {lead['llr_score']})")
        print(f"Type: {lead['type']}")

        # Display the full updated sequence
        seq = lead['sequence']
        # Formatting sequence in blocks of 10 for readability
        formatted_seq = " ".join([seq[i:i+10] for i in range(0, len(seq), 10)])

        print(f"Updated Sequence:\n{formatted_seq}")


    # 5. GLOBAL BIBLIOGRAPHY
    print(f"\n[SECTION 8: BIBLIOGRAPHY & CITATIONS]")
    refs = state.get("references", [])
    if refs:
        # Deduplicate and print
        for j, ref in enumerate(list(dict.fromkeys(refs)), 1):
            print(f"   [{j}] {ref}")
    else:
        print("   No explicit references cited.")


    print("\n" + "═"*80)
    return state

#working

from datetime import datetime # Add this import
import json


def json_export_node(state: MasterState):
    print("--- Node: Exporting Research Data to JSON ---")

    # Structure the data for archival
    export_data = {
        "metadata": {
            # Now datetime will be recognized
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "benchmark_index": state.get("benchmark_index"),  # which of the 9 prompts (None if single-run mode)
            "repeat_index": state.get("repeat_index"),  # which repeat of that prompt (None if single-run mode)
            "iterations": state.get("iteration_count", 0),
            "target_antigen": state.get("target_antigen")
        },
        "clinical_research": {
            "patient_input": state.get("patient_description"),
            "initial_draft": state.get("initial_analysis"),
            "final_revised_analysis": state.get("clinical_analysis")
        },
        "biological_phase": {
            "target": state.get("target_antigen"),
            "antigen_sequence": state.get("antigen_sequence"),
            # FIX: was state.get("antigen_evidence") — that key never existed;
            # the actual field set by antigen_sequence_node is antigen_evidence_url.
            "antigen_evidence_url": state.get("antigen_evidence_url"),
            # FIX: previously just the bare list of antibody names
            # (state.get("antibody_leads")), which itself was always empty
            # before lead_extraction_node existed. Now exports the full
            # record per antibody — name, heavy/light chain sequences, and
            # the retrieval evidence URL — built from the same flat state
            # fields lead_extraction_node populates.
            "discovered_leads": [
                {
                    "name": name,
                    "heavy_chain": state.get("heavy_chains", {}).get(name),
                    "light_chain": state.get("light_chains", {}).get(name),
                    "evidence_url": state.get("ab_sequence_evidence_urls", {}).get(name),
                }
                for name in state.get("antibody_leads", [])
            ],
            # FIX: was state.get("adc_details") — that key was never set
            # anywhere; the real ADC records live in found_adcs.
            "adc_configurations": state.get("found_adcs"),
            # NEW: chains that were retrieved but rejected by the validation
            # gate in lead_extraction_node (e.g. fabricated antigen-as-
            # antibody substitutions) -- kept here for transparency so a
            # "null" outcome in the exported leads can be distinguished from
            # "never attempted" versus "attempted, caught, and rejected."
            "rejected_leads": state.get("rejected_leads", [])
        },
        "engineering_phase": {
            "structural_masking": "Chothia / ANARCI",
            "model_used": "ESM-1v (650M Parameters)",
            "hotspot_data": state.get("hotspot_reports"),   # Now keyed by antibody name: {name: {heavy_hotspots, light_hotspots}}
            "engineered_candidates": state.get("optimized_sequences") # List across ALL antibodies: {id, antibody_name, sequence, llr_score, pos_idx, wt, mut, chain}
        },
        "bibliographic_references": state.get("references", [])
    }

    # FIX: filename now leads with a zero-padded benchmark index (e.g. "01_")
    # AND a repeat suffix (e.g. "_rep03") when running the 9-prompt benchmark
    # with multiple repeats per prompt, so files sort in prompt order and
    # never collide even if two prompts share an antigen or two repeats
    # finish within the same second. Falls back to no prefix/suffix in
    # single-run mode.
    antigen_name = str(state.get("target_antigen", "unknown")).replace("/", "_")
    bidx = state.get("benchmark_index")
    ridx = state.get("repeat_index")
    prefix = f"{bidx+1:02d}_" if isinstance(bidx, int) else ""
    rep_suffix = f"_rep{ridx+1:02d}" if isinstance(ridx, int) else ""
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"ADC_Report_{prefix}{antigen_name}{rep_suffix}_{run_timestamp}.json"

    with open(filename, "w") as f:
        json.dump(export_data, f, indent=4)

    print(f"✅ Research data archived in: {filename}")

    # NEW: per-run human-readable summary, saved alongside the JSON with the
    # same prefix/antigen/repeat/timestamp so the two files pair up 1:1 --
    # same idea as the CSV that used to be written here, but a text report
    # instead. Self-contained (see module docstring) -- does not import
    # benchmark_analysis.py, which stays a fully independent standalone tool.
    summary_filename = f"ADC_RunSummary_{prefix}{antigen_name}{rep_suffix}_{run_timestamp}.txt"
    run_summary_text = _render_run_summary(export_data, filename_label=filename)
    with open(summary_filename, "w", encoding="utf-8") as f:
        f.write(run_summary_text)

    print(f"✅ Run summary archived in: {summary_filename}")

    return state
