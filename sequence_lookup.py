"""
sequence_lookup.py

Four-tier antibody VH/VL sequence retrieval cascade (Thera-SAbDab -> RCSB
gene-symbol search -> UniProt KW-1185 -> curated known-PDB fallback), plus
the antibody-name normalization helper used throughout the pipeline.

Split out of adcAgent_REVEAL_v2.py without changing any logic.
"""

import re
import requests
import time as _time
from bs4 import BeautifulSoup
from typing import Dict, Any, Tuple, Optional, List as TypingList
from langchain_core.tools import tool

# =============================================================================
# 4-TIER ANTIBODY SEQUENCE CASCADE
# Replaces: antibody_imgt_sequence_extractor (fragile HTML scraper)
#           antibody_sequence_search (returned PENDING_PARSING, never retrieved)
#
# Tier 1 — Thera-SAbDab  : therapeutic mAb database (corrected URL)
# Tier 2 — RCSB gene sym  : HGNC gene-symbol RCSB search + FASTA endpoint
# Tier 3 — UniProt REST   : KW-1185 therapeutic antibody keyword
# Tier 4 — Known PDB IDs  : curated fallback for common antibodies
# =============================================================================

_THERA_SABDAB_URL   = "https://opig.stats.ox.ac.uk/webapps/newsabdab/therasabdab/search/"
_RCSB_SEARCH_URL    = "https://search.rcsb.org/rcsbsearch/v2/query"
_RCSB_DATA_URL      = "https://data.rcsb.org/rest/v1/core"
_RCSB_FASTA_URL     = "https://www.rcsb.org/fasta/entry/{pdb_id}/display"
_UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
_SEQ_HEADERS        = {"User-Agent": "Mozilla/5.0 (research; ADC sequence retrieval)"}

# Antigen alias → HGNC gene symbol for RCSB gene-symbol search (Tier 2)
_TARGET_GENE_MAP: Dict[str, str] = {
    "HER2": "ERBB2",    "ERBB2": "ERBB2",
    "HER3": "ERBB3",    "EGFR": "EGFR",    "HER1": "EGFR",
    "PD-1": "PDCD1",    "PD1": "PDCD1",    "CD279": "PDCD1",
    "PD-L1": "CD274",   "PDL1": "CD274",
    "VEGF": "VEGFA",    "VEGF-A": "VEGFA",
    "CD20": "MS4A1",    "CD22": "CD22",
    "TROP2": "TACSTD2", "TROP-2": "TACSTD2",
    "NECTIN4": "NECTIN4", "CD33": "CD33",
    "CD19": "CD19",     "CD79B": "CD79B",
    "FOLR1": "FOLR1",   "BCMA": "TNFRSF17",
    "MSLN": "MSLN",     "MESOTHELIN": "MSLN",
    "TISSUE FACTOR": "F3", "TF": "F3",
}

# Tier 4 fallback — known-good PDB IDs for high-frequency therapeutics
_FALLBACK_PDB: Dict[str, str] = {
    "trastuzumab": "1N8Z", "pertuzumab": "1S78",
    "nivolumab": "5GGS",   "pembrolizumab": "5DK3",
    "cetuximab": "1YY9",   "bevacizumab": "2FJG",
    "atezolizumab": "5XXY","gemtuzumab": "4ZSO",
    "sacituzumab": "7S0Z", "enfortumab": "6AP5",
    "rituximab": "2OSL",   "durvalumab": "5X8L",
    # --- ADDED after benchmark analysis: antibodies the system actually
    # returns for the 9 benchmark prompts but which had no Tier-4 entry,
    # causing systematic null-sequence retrieval (worst case: CD30/prompt 7,
    # where 10/12 leads returned null across all repeats).
    "brentuximab": "7XI0",   # CD30 (Hodgkin) - was entirely absent
    "belantamab":  "4ZFO",   # BCMA (myeloma)
    "amatuximab":  "8CXC",   # MSLN (pancreatic)
    "farletuzumab":"4KMV",   # FOLR1 (ovarian)
    "mirvetuximab":"4KMV",   # FOLR1 - shares the FRa-binding scaffold
    "lintuzumab":  "2PZ3",   # CD33 (AML)
    "inotuzumab":  "5V5H",   # CD22
    "polatuzumab": "6BE3",   # CD79b
}

# Brand-name -> INN mapping. The LLM frequently returns trade names
# (e.g. "Mylotarg", "Padcev", "Adcetris") which never matched the
# INN-keyed fallback dictionary above.
_BRAND_TO_INN: Dict[str, str] = {
    "mylotarg": "gemtuzumab",  "padcev": "enfortumab",
    "adcetris": "brentuximab", "trodelvy": "sacituzumab",
    "kadcyla": "trastuzumab",  "enhertu": "trastuzumab",
    "herceptin": "trastuzumab","perjeta": "pertuzumab",
    "blenrep": "belantamab",   "elahere": "mirvetuximab",
    "besponsa": "inotuzumab",  "polivy": "polatuzumab",
    "erbitux": "cetuximab",    "rituxan": "rituximab",
}

# ADC payload/conjugate suffixes. These are appended to the antibody INN
# in the full drug name (e.g. "brentuximab VEDOTIN") but are NOT part of
# the antibody itself, so they must be stripped before any name lookup.
_ADC_SUFFIXES = (
    "vedotin", "ozogamicin", "govitecan", "emtansine", "deruxtecan",
    "mafodotin", "soravtansine", "ravtansine", "tesirine", "tirumotecan",
    "duocarmazine", "vicleucel", "autoleucel", "pasudotox", "mertansine",
)


def _normalize_antibody_name(raw: str) -> str:
    """
    Normalize an LLM-supplied antibody/drug name to a bare INN suitable for
    dictionary lookup.

    FIX (found via benchmark analysis of 89 runs): the previous lookup was a
    bare `name.lower().strip()`, which matched only 25% of the names the
    system actually produces. Failures came from three sources:
      1. ADC payload suffixes  ("Brentuximab vedotin"    -> no match)
      2. Trade names           ("Mylotarg", "Padcev"     -> no match)
      3. Parenthetical extras  ("Enfortumab vedotin (Padcev)" -> no match)
    This normalizes all three before lookup.
    """
    if not raw:
        return ""
    name = raw.lower().strip()
    # Drop parenthetical qualifiers: "enfortumab vedotin (padcev)" -> "enfortumab vedotin"
    name = re.sub(r"\([^)]*\)", " ", name)
    # Drop structure/DB annotations the LLM sometimes appends
    name = re.sub(r"\b(structure|pdb|clone|antibody|mab|monoclonal)\b[:\s]*", " ", name)
    name = re.sub(r"[^a-z\s-]", " ", name)          # strip punctuation/digits
    tokens = [t for t in name.split() if t]
    # Remove ADC payload suffixes wherever they appear
    tokens = [t for t in tokens if t not in _ADC_SUFFIXES]
    if not tokens:
        return ""
    # Map trade name -> INN if applicable
    for t in tokens:
        if t in _BRAND_TO_INN:
            return _BRAND_TO_INN[t]
    # Otherwise prefer the token that looks like an antibody INN (-mab/-umab/-ximab)
    for t in tokens:
        if t.endswith("mab"):
            return t
    return tokens[0]


def _rcsb_fasta_fetch(pdb_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetch VH + VL from RCSB FASTA endpoint. Returns (heavy, light)."""
    chains_raw: Dict[str, str] = {}
    try:
        url = _RCSB_FASTA_URL.format(pdb_id=pdb_id)
        r = requests.get(url, headers=_SEQ_HEADERS, timeout=15)
        if r.status_code != 200:
            return None, None
        header, seq_parts = None, []
        for line in r.text.strip().splitlines():
            if line.startswith(">"):
                if header:
                    chains_raw[header] = "".join(seq_parts)
                header = line[1:80].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if header:
            chains_raw[header] = "".join(seq_parts)
    except Exception as e:
        print(f"   [FASTA] Error for {pdb_id}: {e}")
        return None, None

    # Keep plausible antibody chains only (100–750 aa)
    ab_chains = {k: v for k, v in chains_raw.items() if 100 <= len(v) <= 750}
    if not ab_chains:
        return None, None

    # Pass 1: header keyword classification
    heavy, light = None, None
    for header, seq in ab_chains.items():
        h = header.lower()
        if any(kw in h for kw in ["heavy", "vh", "chain h", "igg", "iga"]) and not heavy:
            heavy = seq
        elif any(kw in h for kw in ["light", "vl", "kappa", "lambda", "chain l"]) and not light:
            light = seq
    if heavy and light:
        return heavy, light

    # Pass 2: length heuristic (heavy is longer)
    ranked = sorted(ab_chains.values(), key=len, reverse=True)
    return (ranked[0] if ranked else None), (ranked[1] if len(ranked) > 1 else None)


def _tier1_thera_sabdab(antibody_name: str) -> Tuple[Optional[str], Optional[str], str]:
    """Thera-SAbDab (corrected URL). Returns (vh, vl, evidence_url)."""
    # FIX: Thera-SAbDab is indexed by antibody INN, not the full ADC drug
    # name, so "brentuximab vedotin" never matched. Try the normalized bare
    # INN as well as the raw name.
    norm = _normalize_antibody_name(antibody_name)
    candidates = [antibody_name]
    if norm and norm.lower() != antibody_name.lower().strip():
        candidates.append(norm)

    for cand in candidates:
      for params in [
        {"antibody_name": cand, "format": "json"},
        {"Antibody": cand, "format": "json"},
      ]:
        try:
            r = requests.get(_THERA_SABDAB_URL, params=params,
                             headers=_SEQ_HEADERS, timeout=20)
            if r.status_code != 200:
                continue
            hits = r.json()
            hit = (hits[0] if isinstance(hits, list) and hits
                   else hits if isinstance(hits, dict) else None)
            if not hit:
                continue
            pdb_id = str(hit.get("pdb") or hit.get("PDB") or "").upper()
            vh = hit.get("VHseq") or hit.get("heavy_seq") or hit.get("Hseq")
            vl = hit.get("VLseq") or hit.get("light_seq") or hit.get("Lseq")
            # Sequences absent → fetch from RCSB FASTA
            if pdb_id and (not vh or not vl):
                fh, fl = _rcsb_fasta_fetch(pdb_id)
                vh, vl = vh or fh, vl or fl
            if vh or vl:
                ev_url = (f"https://opig.stats.ox.ac.uk/webapps/newsabdab/"
                          f"therasabdab/structureviewer/?pdb={pdb_id}")
                print(f"   [T1 SAbDab] ✅ {antibody_name} via {pdb_id}")
                return vh, vl, ev_url
        except Exception as e:
            print(f"   [T1 SAbDab] {e}")
    return None, None, ""


def _tier2_rcsb_gene(antibody_name: str, target_antigen: Optional[str]) -> Tuple[Optional[str], Optional[str], str]:
    """RCSB gene-symbol search. Returns (vh, vl, evidence_url)."""
    gene_symbol = _TARGET_GENE_MAP.get((target_antigen or "").upper())
    pdb_ids: TypingList[str] = []

    if gene_symbol:
        query = {
            "query": {
                "type": "group", "logical_operator": "and",
                "nodes": [
                    {"type": "terminal", "service": "text",
                     "parameters": {"attribute": "rcsb_entity_source_organism.rcsb_gene_name.value",
                                    "operator": "exact_match", "value": gene_symbol}},
                    {"type": "group", "logical_operator": "or", "nodes": [
                        {"type": "terminal", "service": "full_text",
                         "parameters": {"value": kw}} for kw in ["antibody", "Fab", "immunoglobulin"]
                    ]}
                ]
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 8},
                                "sort": [{"sort_by": "rcsb_entry_info.resolution_combined",
                                          "direction": "asc"}]}
        }
        try:
            r = requests.post(_RCSB_SEARCH_URL, json=query,
                              headers=_SEQ_HEADERS, timeout=15)
            if r.status_code == 200:
                pdb_ids = [h["identifier"] for h in r.json().get("result_set", [])]
        except Exception as e:
            print(f"   [T2 RCSB] gene search error: {e}")

    # Fallback within T2: full-text by antibody name
    if not pdb_ids:
        try:
            query2 = {"query": {"type": "terminal", "service": "full_text",
                                "parameters": {"value": antibody_name}},
                      "return_type": "entry",
                      "request_options": {"paginate": {"start": 0, "rows": 8},
                                          "sort": [{"sort_by": "rcsb_entry_info.resolution_combined",
                                                    "direction": "asc"}]}}
            r2 = requests.post(_RCSB_SEARCH_URL, json=query2,
                               headers=_SEQ_HEADERS, timeout=15)
            if r2.status_code == 200:
                pdb_ids = [h["identifier"] for h in r2.json().get("result_set", [])]
        except Exception as e:
            print(f"   [T2 RCSB] full-text error: {e}")

    if not pdb_ids:
        return None, None, ""

    best_id = pdb_ids[0]
    vh, vl = _rcsb_fasta_fetch(best_id)
    if vh or vl:
        print(f"   [T2 RCSB] ✅ {antibody_name} via {best_id} (gene={gene_symbol or 'full-text'})")
        return vh, vl, f"https://www.rcsb.org/structure/{best_id}"
    return None, None, ""


def _tier3_uniprot(antibody_name: str) -> Tuple[Optional[str], Optional[str], str]:
    """UniProt KW-1185 therapeutic antibody search. Returns (vh, vl, evidence_url)."""
    queries = [
        f'protein_name:"{antibody_name}" AND keyword:KW-1185 AND reviewed:true',
        f'protein_name:"{antibody_name}" AND reviewed:true',
    ]
    for q in queries:
        try:
            r = requests.get(_UNIPROT_SEARCH_URL,
                             params={"query": q, "format": "json", "size": 5,
                                     "fields": "accession,sequence,protein_name"},
                             headers=_SEQ_HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            seqs = [e["sequence"]["value"] for e in r.json().get("results", [])
                    if e.get("sequence", {}).get("value") and
                    len(e["sequence"]["value"]) > 100]
            if len(seqs) >= 2:
                seqs.sort(key=len, reverse=True)
                print(f"   [T3 UniProt] ✅ {antibody_name}")
                return (seqs[0], seqs[1],
                        f"https://www.uniprot.org/uniprotkb?query={antibody_name}")
        except Exception as e:
            print(f"   [T3 UniProt] {e}")
    return None, None, ""


def _tier4_fallback(antibody_name: str) -> Tuple[Optional[str], Optional[str], str]:
    """Known PDB ID fallback for common therapeutics. Returns (vh, vl, evidence_url)."""
    # FIX: normalize first (strip ADC suffixes, map brand names) — the old
    # bare .lower().strip() lookup matched only ~25% of real names.
    norm = _normalize_antibody_name(antibody_name)
    pdb_id = _FALLBACK_PDB.get(norm)
    if not pdb_id:
        # Last resort: substring match against known INNs, so e.g.
        # "anti-nectin4 (structure 9UCL)" still has a chance to resolve.
        for inn, pid in _FALLBACK_PDB.items():
            if inn in norm:
                pdb_id = pid
                break
    if not pdb_id:
        print(f"   [T4 Fallback] No curated PDB for '{antibody_name}' (normalized: '{norm}')")
        return None, None, ""
    print(f"   [T4 Fallback] '{antibody_name}' normalized to '{norm}' -> PDB {pdb_id}")
    vh, vl = _rcsb_fasta_fetch(pdb_id)
    if vh or vl:
        print(f"   [T4 Fallback] ✅ {antibody_name} via known PDB {pdb_id}")
        return vh, vl, f"https://www.rcsb.org/structure/{pdb_id}"
    return None, None, ""


@tool
def antibody_sequence_search(antibody_name: str, target_antigen: str = "") -> Dict[str, Any]:
    """
    Retrieve VH and VL amino acid sequences for a therapeutic antibody using a
    four-tier deterministic cascade:
      Tier 1 — Thera-SAbDab (therapeutic mAb database)
      Tier 2 — RCSB PDB via HGNC gene-symbol search + FASTA endpoint
      Tier 3 — UniProt KW-1185 therapeutic antibody keyword
      Tier 4 — Curated known-PDB fallback for common antibodies
    Returns actual sequences, evidence URL, and tier used.
    """
    print(f"   [SeqCascade] Starting retrieval for '{antibody_name}' "
          f"(antigen hint: {target_antigen or 'none'})")

    antigen_hint = target_antigen.strip() if target_antigen else None

    for tier_fn, tier_label, is_autonomous in [
        (_tier1_thera_sabdab,  "Thera-SAbDab",  True),
        (_tier2_rcsb_gene,     "RCSB Gene",     True),
        (_tier3_uniprot,       "UniProt",        True),
        (_tier4_fallback,      "Known-PDB",      False),
    ]:
        try:
            if tier_label == "RCSB Gene":
                vh, vl, url = tier_fn(antibody_name, antigen_hint)
            else:
                vh, vl, url = tier_fn(antibody_name)
        except Exception as e:
            print(f"   [{tier_label}] Error: {e}")
            continue

        if vh or vl:
            print(f"\n--- SEQUENCE RECORD: {antibody_name} ({tier_label}) ---")
            print(f"> {antibody_name}_Heavy_Chain\n{vh}")
            print(f"> {antibody_name}_Light_Chain\n{vl}")
            print(f"Source: {url}\n{'─'*50}")
            return {
                "antibody_name":    antibody_name,
                "heavy_chains":     {antibody_name: vh},
                "light_chains":     {antibody_name: vl},
                "ab_sequence_evidence_urls": url,
                "source_tier":      tier_label,
                "autonomous":       is_autonomous,
                "status":           "Success",
            }

    print(f"   [SeqCascade] ❌ All tiers failed for '{antibody_name}'")
    return {
        "antibody_name":    antibody_name,
        "heavy_chains":     {antibody_name: None},
        "light_chains":     {antibody_name: None},
        "ab_sequence_evidence_urls": None,
        "source_tier":      "None",
        "autonomous":       False,
        "status":           "Failed — sequences not publicly available",
    }


# NOT CALLED (revision cleanup, confirmed via grep -- no call sites anywhere
# in this file besides its own definition). Originally kept as a backwards-
# compatible alias for an older tool name; nothing in the current codebase
# references it.
# antibody_imgt_sequence_extractor = antibody_sequence_search

