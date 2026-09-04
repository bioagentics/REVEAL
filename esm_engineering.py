"""
esm_engineering.py

Sequence engineering phase: ESM-1v model loading/caching, CDR index
extraction, the per-chain LLR scan, and the two nodes that turn LLR
hotspots into engineered candidate sequences.

Note: ESM-1v LLR is an evolutionary-plausibility signal, not a binding-
affinity predictor -- see the paper's Limitations section.

Split out of adcAgent_REVEAL_v2.py without changing any logic.
"""

from state import MasterState
from config import anarci

import torch
import esm

# --- GLOBAL MODEL INITIALIZATION ---
# This ensures the model stays in memory and isn't re-initialized by the node.
MODEL_CACHE = {}

def get_esm_resources():
    """Accesses the global model components, loading them only if necessary."""
    if "model" not in MODEL_CACHE:
        print("--- Loading ESM-1v (Optimized Global Load) ---")

        # Load model and alphabet
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()
            print("Model moved to GPU (CUDA).")
        else:
            print("CUDA not available, using CPU RAM.")

        MODEL_CACHE["model"] = model
        MODEL_CACHE["alphabet"] = alphabet
        MODEL_CACHE["converter"] = alphabet.get_batch_converter()

    # FIX: previously only (model, converter) were returned, but run_chain_scan
    # also needs `alphabet` for get_idx(). Downstream code referenced bare
    # globals MODEL / ALPHABET / BATCH_CONVERTER that were never assigned
    # anywhere, so hotspot scanning silently no-op'd via the outer try/except.
    return MODEL_CACHE["model"], MODEL_CACHE["alphabet"], MODEL_CACHE["converter"]

# Pre-load immediately at script start to catch memory issues early

#working
def get_cdr_indices_robust(sequence, chain_type):
    """Robust CDR identification using Chothia numbering."""
    print("Robust CDR identification using Chothia numbering")

    # FIX (found via actual run logs across all 9 benchmark prompts):
    # `import anarci` above imports the MODULE, not the callable. Calling
    # anarci(...) directly raises "'module' object is not callable" and was
    # being silently swallowed by the outer try/except in every single run,
    # which is why sequences were retrieved correctly but Section 6/7
    # (hotspots / engineered sequences) always came back empty. The actual
    # callable lives at anarci.anarci(...).
    if anarci is None:
        print(" [!] ANARCI module not available — cannot identify CDRs.")
        return []

    anarci_res = anarci.anarci([("query", sequence)], scheme="chothia", output=False)
    if not anarci_res or anarci_res[0] is None:
        return []

    def flatten_to_tuples(iterable):
        for item in iterable:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], str) and len(item[1]) == 1:
                # This looks like ((numbering), 'A')
                yield item
            elif isinstance(item, (list, tuple)):
                yield from flatten_to_tuples(item)
    # FIX: was referencing an undefined variable `results` (typo for anarci_res),
    # which raised a NameError caught silently by the caller's try/except,
    # meaning CDR indices (and therefore all hotspots) were always empty.
    extracted = list(flatten_to_tuples(anarci_res))

    indices = []
    seq_counter = 0

    # FIX: chain_type was accepted as a parameter but never used — CDR ranges
    # were hardcoded to heavy-chain Chothia numbering for both chains, so
    # light-chain CDR loops (which fall in different position ranges) were
    # never correctly identified. Standard Chothia CDR definitions:
    #   Heavy: CDR-H1 26-32, CDR-H2 52-56, CDR-H3 95-102
    #   Light: CDR-L1 24-34, CDR-L2 50-56, CDR-L3 89-97
    if chain_type == "L":
        cdr_ranges = [(24, 34), (50, 56), (89, 97)]
    else:
        cdr_ranges = [(26, 32), (52, 56), (95, 102)]

    for entry in extracted:
        pos_data, aa = entry

        if aa == '-':
            continue

        # Get the integer from the pos_data (which could be 26 or (26, ' '))
        try:
            if isinstance(pos_data, (list, tuple)):
                res_num = int(pos_data[0])
            else:
                res_num = int(pos_data)

            if any(lo <= res_num <= hi for lo, hi in cdr_ranges):
                indices.append(seq_counter)

            seq_counter += 1
        except:
            continue

    print(f"Success! Found {len(indices)} CDR residues.")
    # FIX: this was `print("Indices identified:", results)` — a second
    # reference to the same undefined variable that caused the original bug.
    # This would have raised a NameError immediately after computing indices,
    # crashing the function again even with the anarci.anarci() call fixed.
    print("Indices identified:", indices)
    return indices

#working
def run_chain_scan(sequence, chain_type, llr_threshold: float = 0.0):
    """
    Internal helper to identify CDRs and calculate LLR scores.

    Scoring strategy (documented for reproducibility, per revision review):
    This is a WT-MARGINAL scan — a single forward pass is run on the wild-type
    sequence, and each candidate mutation's log-likelihood is read off the same
    logits tensor (no masking, no mutant-sequence re-encoding). This is faster
    than masked-marginal scoring but is known to be less accurate for large
    log-odds shifts (Meier et al. 2021). State this explicitly in the paper's
    Methods/Reproducibility section rather than leaving the scoring convention
    implicit.
    """
    print("Internal helper to identify CDRs and calculate LLR scores.")
    # 1. Identify CDRs using the nested-fix logic
    cdr_indices = get_cdr_indices_robust(sequence, chain_type)

    print("sequence: ",sequence )
    print(f"Scanning {len(cdr_indices)} residues for hotspots...")

    # 2. ESM-1v Inference
    # FIX: fetch resources directly rather than relying on bare globals
    # (MODEL / ALPHABET / BATCH_CONVERTER) that were never assigned anywhere,
    # which previously caused this function to raise NameError on every call.
    model, alphabet, batch_converter = get_esm_resources()

    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    if torch.cuda.is_available(): batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        output = model(batch_tokens, repr_layers=[33])
        logits = output["logits"]

    hotspots = []
    for i in cdr_indices:
        wt_aa = sequence[i]
        wt_idx = alphabet.get_idx(wt_aa)
        wt_log_prob = logits[0, i + 1, wt_idx].item()

        for aa in "ACDEFGHIKLMNPQRSTVWY":
            if aa == wt_aa: continue
            mut_idx = alphabet.get_idx(aa)
            mut_log_prob = logits[0, i + 1, mut_idx].item()
            llr = mut_log_prob - wt_log_prob
            #print("aa",aa, "wt_aa", wt_aa, "mut_log_prob",mut_log_prob, "LLR:", llr )

            # NOTE: llr_threshold defaults to 0.0 to preserve prior behavior
            # ("LLR > 0" per Eq. 1-2 in the paper). Pass a higher threshold
            # here if the revision moves to a stricter hotspot-confidence cut.
            if llr > llr_threshold:
                hotspots.append({
                    "chain": chain_type,
                    "pos_idx": i,
                    "wt": wt_aa,
                    "mut": aa,
                    "llr": round(llr, 3)
                })
    #print("hotspots:", hotspots)
    return sorted(hotspots, key=lambda x: x['llr'], reverse=True)

def clean_seq(raw):
    """
    Safely extracts a sequence string from either a bare string or a
    {antibody_name: sequence} dict (the actual shape heavy_chains/
    light_chains are stored in). Shared by antibody_optimization_node and
    mutation_application_node.
    """
    if isinstance(raw, dict) and raw:
        # Taking the first value found in the dictionary
        seq = next(iter(raw.values()))
    else:
        seq = raw
    return seq.strip().upper() if isinstance(seq, str) and len(seq) > 10 else None


def antibody_optimization_node(state: MasterState):
    """
    Step 2: Affinity Maturation Node.
    Identifies mutation hotspots in CDR loops using ESM-1v.

    FIX: previously used clean_seq() to pull out only next(iter(raw.values())) —
    the FIRST antibody in the heavy_chains/light_chains dicts — so when
    multiple antibodies were discovered (e.g. Trastuzumab AND Pertuzumab),
    only the first one ever got scanned; the rest were silently dropped from
    the entire engineering phase. This now loops over every antibody name
    present in either dict and scans each one independently.
    """
    heavy_map = state.get("heavy_chains", {}) or {}
    light_map = state.get("light_chains", {}) or {}

    print("\n--- [NODE] Starting Hotspot Identification ---")

    # Union of every antibody name that has at least one chain available.
    all_names = sorted(set(heavy_map.keys()) | set(light_map.keys()))

    # Keep only names with a usable (long-enough) sequence on at least one chain.
    def usable(name):
        vh = heavy_map.get(name)
        vl = light_map.get(name)
        return (isinstance(vh, str) and len(vh) > 10) or (isinstance(vl, str) and len(vl) > 10)

    all_names = [n for n in all_names if usable(n)]

    reports_by_antibody = {}

    if not all_names:
        # REMOVED (revision cleanup): this used to silently substitute a
        # hardcoded Trastuzumab Fab sequence and engineer mutations on it
        # whenever no real antibody was retrieved for the prompt, reporting
        # those mutations as if they belonged to the actual query. That
        # directly reproduces the paper's own documented limitation
        # ("relies on the default sequence of Trastuzumab for successful
        # execution") and would silently contaminate the benchmark re-run —
        # a prompt that genuinely failed retrieval would still show up as an
        # "engineered" success. Since the paper's contribution is now the
        # honest characterization of failure modes, this node must fail
        # loud (empty result) instead of substituting an unrelated antibody.
        print(" [!] No valid sequences found for this prompt — skipping "
              "engineering phase. This run should be counted as a "
              "retrieval/engineering failure, not silently backfilled.")
        return {"hotspot_reports": {}}

    print("--- Sequences detected. Loading ESM-1v (Optimized Global Load) ---")
    _ = get_esm_resources()

    for name in all_names:
        vh_seq = clean_seq(heavy_map.get(name))
        vl_seq = clean_seq(light_map.get(name))

        reports = {"heavy_hotspots": [], "light_hotspots": []}

        if vh_seq:
            print(f"[{name}] Scanning Heavy Chain ({len(vh_seq)} AA)...")
            try:
                reports["heavy_hotspots"] = run_chain_scan(vh_seq, chain_type="H")
            except Exception as e:
                print(f" [!] Error scanning Heavy Chain for {name}: {e}")

        if vl_seq:
            print(f"[{name}] Scanning Light Chain ({len(vl_seq)} AA)...")
            try:
                reports["light_hotspots"] = run_chain_scan(vl_seq, chain_type="L")
            except Exception as e:
                print(f" [!] Error scanning Light Chain for {name}: {e}")

        reports_by_antibody[name] = reports

    print(f"[LeadEngineering] Scanned {len(reports_by_antibody)} antibody/antibodies: {list(reports_by_antibody.keys())}")
    return {"hotspot_reports": reports_by_antibody}

def mutation_application_node(state: MasterState):
    """
    Step 3: Engineering Node.
    Applies the top-ranked mutations to the wild-type sequences.
    """
    # FIX: hotspot_reports is now keyed by antibody name (see
    # antibody_optimization_node), e.g.
    # {"Trastuzumab": {"heavy_hotspots": [...], "light_hotspots": [...]},
    #  "Pertuzumab":  {"heavy_hotspots": [...], "light_hotspots": [...]}}
    # Previously this function only ever used ONE global vh_wt/vl_wt (pulled
    # from just the first antibody via clean_seq()), so mutations were only
    # ever applied to a single antibody even when several were discovered.
    # This now loops over every antibody that has a hotspot report and
    # applies mutations against THAT antibody's own sequences.
    heavy_map = state.get("heavy_chains", {}) or {}
    light_map = state.get("light_chains", {}) or {}
    reports_by_antibody = state.get("hotspot_reports", {}) or {}

    # REMOVED (revision cleanup): this used to fall back to a hardcoded
    # Trastuzumab Fab sequence (fallback_vh/fallback_vl) whenever an
    # antibody name in hotspot_reports had no matching entry in
    # heavy_chains/light_chains, and would apply that antibody's real
    # hotspot positions to the WRONG (Trastuzumab) sequence -- silently
    # mixing one antibody's mutation coordinates with another antibody's
    # backbone. Per the same reasoning as antibody_optimization_node above,
    # this now skips the antibody outright rather than fabricating a
    # mutated sequence that doesn't correspond to anything actually
    # retrieved for this prompt.

    print("Mutation Report (per antibody)", reports_by_antibody)
    print("\n--- [NODE] Applying Mutations to Generate Optimized Leads ---")

    optimized_leads = []
    skipped_antibodies = []

    for antibody_name, reports in reports_by_antibody.items():
        vh_wt = clean_seq(heavy_map.get(antibody_name))
        vl_wt = clean_seq(light_map.get(antibody_name))

        if not vh_wt and not vl_wt:
            print(f" [!] Skipping '{antibody_name}': no matching wild-type "
                  f"sequence in state (hotspots exist but the sequence they "
                  f"were computed from is missing). Not substituting a "
                  f"fallback sequence -- recording as a gap instead.")
            skipped_antibodies.append(antibody_name)
            continue

        # Process Top 3 Heavy Chain Mutations for this antibody
        h_muts = reports.get("heavy_hotspots", [])[:3] if vh_wt else []
        if reports.get("heavy_hotspots") and not vh_wt:
            print(f" [!] '{antibody_name}' has heavy-chain hotspots but no "
                  f"heavy-chain sequence in state -- skipping heavy mutations "
                  f"for this antibody rather than applying them to a "
                  f"different antibody's sequence.")
        for m in h_muts:
            seq_list = list(vh_wt)
            seq_list[m['pos_idx']] = m['mut']
            new_seq = "".join(seq_list)

            optimized_leads.append({
                "id": f"Opt_{antibody_name}_H_{m['wt']}{m['pos_idx']}{m['mut']}",
                "antibody_name": antibody_name,
                "type": "Heavy_Optimization",
                "sequence": new_seq,
                "llr_score": m['llr'],
                "pos_idx": m['pos_idx'],
                "wt": m['wt'],
                "mut": m['mut'],
                "chain": "H"
            })

        # Process Top 2 Light Chain Mutations for this antibody
        l_muts = reports.get("light_hotspots", [])[:2] if vl_wt else []
        if reports.get("light_hotspots") and not vl_wt:
            print(f" [!] '{antibody_name}' has light-chain hotspots but no "
                  f"light-chain sequence in state -- skipping light mutations "
                  f"for this antibody rather than applying them to a "
                  f"different antibody's sequence.")
        for m in l_muts:
            seq_list = list(vl_wt)
            seq_list[m['pos_idx']] = m['mut']
            new_seq = "".join(seq_list)

            optimized_leads.append({
                "id": f"Opt_{antibody_name}_L_{m['wt']}{m['pos_idx']}{m['mut']}",
                "antibody_name": antibody_name,
                "type": "Light_Optimization",
                "sequence": new_seq,
                "llr_score": m['llr'],
                "pos_idx": m['pos_idx'],
                "wt": m['wt'],
                "mut": m['mut'],
                "chain": "L"
            })

    print(f"[MutationApplication] Produced {len(optimized_leads)} optimized lead(s) "
          f"across {len(reports_by_antibody)} antibody/antibodies.")
    if skipped_antibodies:
        print(f"[MutationApplication] Skipped {len(skipped_antibodies)} "
              f"antibody/antibodies with no matching sequence (recorded as a "
              f"gap, not backfilled): {skipped_antibodies}")
    return {"optimized_sequences": optimized_leads}
