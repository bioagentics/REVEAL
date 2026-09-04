"""
nodes_biology.py

Biology / antibody-discovery phase: ReAct biologist node + revisor loop,
the antibody-list normalizer, the plausibility gate, and lead extraction
(the ANARCI validation gate that turns fabricated antigen-as-antibody
hits into honest nulls).

Split out of adcAgent_REVEAL_v2.py without changing any logic.
"""

import re
from state import MasterState
from config import llm, anarci
from tools import tools_by_name, openai_formatted_tools
from sequence_lookup import antibody_sequence_search

"""Biologist node"""

# CELL 12 — ReAct prompt (system message only)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

biologist_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Biologist Research Assistant using ReAct (Reasoning + Acting).\n\n"

     "INPUT:\n"
     "- A cancer antigen name (plain text, no UniProt ID provided).\n\n"

     "GOAL:\n"
     "- Identify EXISTING antibodies that target this antigen.\n"
     "- Retrieve antibody INFORMATION AND SEQUENCES if they exist.\n\n"

     "STRICT RULES:\n"
     "- Retrieval ONLY. Do NOT design, predict, or generate new antibodies.\n"
     "- Do NOT fabricate sequences under any circumstance.\n"
     "- If antibody sequences are not publicly available, explicitly return null.\n\n"

     "PRIORITY OF SOURCES (in this exact order):\n"
     "1) Antibody / ADC databases (e.g., ADCdb, DrugBank, IMGT references)\n"
     "2) Curated biomedical sources (reviews, clinical trial reports)\n"
     "3) General web literature (only if above are unavailable)\n\n"

     "WHEN ANTIBODIES ARE FOUND:\n"
     "- For each antibody, return:\n"
     "  • Antibody name\n"
     "  • Target antigen\n"
     "  • Evidence URLs\n"
     "  • Antibody sequence(s): heavy chain, light chain (if available)\n"
     "- If only partial sequence data is available, return what is known.\n\n"

     "WHEN SEQUENCES ARE NOT FOUND:\n"
     "- Search again from the databases (IMGT, SAbDab, OAS, ABSD) having antibody in a structure and only extract the sequence of antibody\n"
     "- Try that you find sequences, search IMGT in detail"
     "- Set the sequence field explicitly to null\n"
     "- Add a short explanation (e.g., proprietary / not disclosed)\n\n"

     "OUTPUT FORMAT:\n"
     "- Return ONLY a final JSON object when finished.\n"
     "- No markdown, no explanations outside JSON.\n"
    ),
    MessagesPlaceholder(variable_name="biologist_pad")
])

import json
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

def biologist_node(state: MasterState):
    print(f"--- Node: Biologist Research (Direct Tool Execution) ---")

    # 1. Initialize the internal conversation history
    # We pull the antigen from the MasterState (provided by Dr. M)
    antigen = state.get("target_antigen", "Unknown")

    internal_messages = [
        HumanMessage(content=f"Identify existing antibodies and sequences for {antigen}.")
    ]
     # --- PHASE 1: INITIAL RESPONSE ---
    # We bind your AnswerQuestion tool to the LLM
    biologist_chain = biologist_prompt | llm.bind_tools(openai_formatted_tools)


    for i in range(5):
        # A. CALL THE MODEL (Reasoning)
        # model_react uses your react_prompt and scratch_pad variable
        response = biologist_chain.invoke({"biologist_pad": internal_messages})
        internal_messages.append(response)

        # B. CHECK IF TOOLS ARE NEEDED (The Decision)
        if not response.tool_calls:
            print("   [Biologist] No more tools requested. ")
            break

        # C. EXECUTE TOOLS (Your tool_node logic integrated here)
        print(f"   [Biologist] Executing {len(response.tool_calls)} tool(s)...")

        # This is your specific code snippet adapted for the internal loop
        for tc in response.tool_calls:
            # Invoke the tool from your tools_by_name registry
            tool_result = tools_by_name[tc["name"]].invoke(tc["args"])

            # Append the ToolMessage to our internal history
            internal_messages.append(ToolMessage(
                content=json.dumps(tool_result),
                name=tc["name"],
                tool_call_id=tc["id"]
            ))

    # 3. Final Output Extraction
    # We take the final AI message which should contain the JSON object
    final_output = internal_messages[-1].content
    #print(final_output)
    try:
        antibody_data = json.loads(final_output)
    except:
        # Fallback if the LLM adds extra text or fails JSON formatting
        antibody_data = {"raw_data": final_output}
    print("--- Node: Biologist Research (Direct Tool Execution) --- ANTIBODY_DATA",antibody_data)
    print("Antibody name", antibody_data.get("name"))

    return {
        "found_antibodies": antibody_data,
        "messages": [AIMessage(content=f"Biologist found {len(antibody_data)} lead(s) for {antigen}.")]
    }

biologist_revisor_prompt ="""You are the Senior Biologist Validator and Sequence Auditor.
Your role is to perform a high-rigor audit of antibody data.

INPUT PROVIDED:
1) Target Antigen Name.
2) DRAFT JSON containing identified antibodies.

AUDIT PROTOCOL:
- SEQUENCE VERIFICATION: For every antibody where sequences are 'null', perform a deep-dive search (IMGT/mAb-DB, Therapeutic Structural Database, or Patent databases).
- NO FABRICATION: You are strictly forbidden from predicting or generating sequences. If a sequence is not in the public domain after exhaustive search, it MUST remain null.
- DATA INTEGRITY: Ensure 'heavy_chain' and 'light_chain' are valid amino acid strings (A, C, D, E, etc.). Remove any non-biological characters.
- CITATION REQUIREMENT: Every added sequence MUST be accompanied by a specific 'evidence_url'.

DETERMINISTIC BEHAVIOR:
- If the DRAFT JSON is technically complete and no new sequences can be found, return the ORIGINAL JSON exactly as it is.
- Return ONLY the JSON object. Do not include markdown code blocks (```json) or conversational text.

SCHEMA:
{{
  "molecular_design": {{
    "antibody_leads": {{
      "antibodies": [
        {{
          "name": "string",
          "target": "string",
          "heavy_chain": "string|null",
          "light_chain": "string|null",
          "evidence_url": "string",
          "note": "string"
        }}
      ]
    }}
  }}
}}
"""
biologist_revisor_prompt = ChatPromptTemplate.from_messages([
    ("system", biologist_revisor_prompt), # The strict instructions we wrote
    MessagesPlaceholder(variable_name="biologist_pad"), # Where the history goes
])

# NOT CALLED (revision cleanup, confirmed via grep -- no call sites anywhere
# in this file besides its own definition). Was already marked deprecated
# in its own docstring, delegating to the 4-tier cascade below, but nothing
# in the current graph actually invokes it.
# def extract_full_antibody_sequences(antibody_name):
#     """
#     Deprecated. Delegates to the 4-tier cascade (antibody_sequence_search).
#     Returns {heavy, light, url} for backwards compatibility.
#     """
#     result = antibody_sequence_search.invoke(
#         {"antibody_name": antibody_name, "target_antigen": ""}
#     )
#     if result.get("status") == "Success":
#         return {
#             "heavy": result["heavy_chains"].get(antibody_name),
#             "light": result["light_chains"].get(antibody_name),
#             "url":   result.get("ab_sequence_evidence_urls", ""),
#         }
#     return None

def normalize_antibody_list(data):
    """
    Flattens the LLM's antibody JSON into a plain list of antibody dicts,
    regardless of whether it came back as a flat list, a flat name->info
    dict, or wrapped in the nested schema the Biologist Revisor prompt
    itself requests: {"molecular_design": {"antibody_leads": {"antibodies": [...]}}}.
    Shared by biologist_revisor_node and lead_extraction_node.
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        # Common wrapper keys the LLM tends to use, checked in order.
        for key in ("antibodies", "antibody_leads", "molecular_design"):
            if key in data:
                nested = normalize_antibody_list(data[key])
                if nested:
                    return nested

        # If every value looks like an antibody record itself (has a
        # heavy/light chain key), treat this as a flat name->info dict
        # and inject "name" from the key so downstream code still works.
        if all(isinstance(v, dict) for v in data.values()) and data:
            out = []
            for k, v in data.items():
                v = dict(v)
                v.setdefault("name", k)
                out.append(v)
            if out:
                return out

        # Last resort: recursively search all values for a usable list.
        for v in data.values():
            nested = normalize_antibody_list(v)
            if nested:
                return nested

    return []


#updates revisor
def biologist_revisor_node(state: MasterState):
    current_iter = state.get("biologist_iterations", 0)
    print(f"--- Node: Biologist Revisor (Sequence Audit Pass {current_iter + 1}) ---")

    antigen = state.get("target_antigen", "Unknown")
    found_abs = state.get("found_antibodies", {})
    draft_json = json.dumps(found_abs, indent=2)

    # 1. LLM Tool-Based Search (Existing Logic)
    internal_messages = [
        HumanMessage(content=(
            f"Audit this draft JSON for {antigen}. Find missing heavy/light sequences.\n"
            f"DRAFT_DATA: {draft_json}"
        ))
    ]
    revisor_chain = biologist_revisor_prompt | llm.bind_tools(openai_formatted_tools)

    for i in range(3):
        res = revisor_chain.invoke({"biologist_pad": internal_messages})
        internal_messages.append(res)
        if not res.tool_calls: break
        for tc in res.tool_calls:
            tool_result = tools_by_name[tc["name"]].invoke(tc["args"])
            internal_messages.append(ToolMessage(content=json.dumps(tool_result), name=tc["name"], tool_call_id=tc["id"]))

    # Parse LLM Output
    try:
        clean_json_str = re.sub(r'```json\n?|```', '', internal_messages[-1].content).strip()
        updated_data = json.loads(clean_json_str)
    except:
        updated_data = found_abs

    # 2. EXPLICIT FALLBACK: PDB Search for remaining Nulls
    print("   [Revisor] Checking for remaining null sequences via PDB fallback...")

    # normalize_antibody_list is now defined once at module level (shared
    # with lead_extraction_node) rather than redefined inline here.
    antibody_list = normalize_antibody_list(updated_data)
    if not antibody_list:
        print("   [Revisor] No parseable antibody records found — skipping PDB fallback pass.")

    for item in antibody_list:
        name = item.get("name")

        # Check both singular and plural key variants to be safe
        has_heavy = bool(item.get("heavy_chains") or item.get("heavy_chain"))
        has_light = bool(item.get("light_chains") or item.get("light_chain"))

        if name and (not has_heavy or not has_light):
            print(f"   [Fallback] Running 4-tier cascade for: {name}")
            # Use the same cascade tool (no Tavily, no LLM — deterministic)
            antigen_hint = item.get("target", "")
            seq_result = antibody_sequence_search.invoke(
                {"antibody_name": name, "target_antigen": antigen_hint}
            )
            if seq_result.get("status") == "Success":
                if not has_heavy:
                    item["heavy_chains"] = seq_result["heavy_chains"].get(name)
                if not has_light:
                    item["light_chains"] = seq_result["light_chains"].get(name)
                item["ab_sequence_evidence_urls"] = seq_result.get("ab_sequence_evidence_urls")
                item["source_tier"] = seq_result.get("source_tier")
                print(f"   [Success] Updated '{name}' via {seq_result.get('source_tier')}")


    return {
        "found_antibodies": updated_data,
        "biologist_iterations": current_iter + 1,
        "messages": [AIMessage(content=f"Biologist Revision complete. PDB fallback checked.")]
    }

def should_biologist_revise(state: MasterState) -> str:
    # Rule: Only allow ONE revision pass for the Biologist to prevent infinite sequence searching
    if state.get("biologist_iterations", 0) >= 1:
        return "finalize"

    # Check if there are still null sequences that might be worth one more search
    leads = state.get("found_antibodies", {}).get("molecular_design", {}).get("antibody_leads", {}).get("antibodies", [])
    has_nulls = any(ab.get("heavy_chain") is None for ab in leads)

    if has_nulls:
        return "revise"
    return "finalize"


def _is_plausible_antibody_sequence(seq, max_len=260):
    """
    VALIDATION GATE (added post-benchmark-analysis): checks whether a
    sequence retrieved as an antibody chain is actually plausible as an
    antibody variable domain (VH/VL) or Fab chain, rather than a
    mislabeled antigen fragment.

    ROOT CAUSE this addresses: the documented "fabrication" failure mode
    (see benchmark analysis, FOLR1 and CD33 prompts -- 8/25 and 8/14 leads
    respectively) where retrieval fails and the pipeline instead returns a
    substring of the TARGET ANTIGEN's own sequence, mislabeled as if it
    were the antibody. Previously this was only caught downstream, and
    silently, when ANARCI found zero CDR residues in it (hotspot scanning
    just came back empty) -- but the fabricated sequence still got
    exported into discovered_leads as if it were a real, successful hit.

    This runs the same check EARLIER and DECISIVELY: if a sequence cannot
    be numbered as an immunoglobulin V-domain by ANARCI, and/or its length
    is far outside the range of a real VH/VL/Fab chain (~100-230aa; most
    target antigens used in this benchmark are 200aa-1300aa+), it is
    rejected here rather than accepted as a lead. A rejected sequence is
    treated as a genuine retrieval failure (null), which is the honest
    outcome -- NOT silently exported as a successful hit.

    Returns (is_plausible: bool, reason: str).
    """
    if not seq or not isinstance(seq, str) or len(seq) < 10:
        return False, "sequence missing or too short to evaluate"

    too_long = len(seq) > max_len

    if anarci is None:
        # ANARCI unavailable in this environment -- fall back to the length
        # heuristic alone. Weaker signal, but still catches the most
        # egregious cases (e.g. a 1255aa antigen substring labeled as a
        # single antibody chain).
        if too_long:
            return False, (f"ANARCI unavailable; sequence length {len(seq)}aa "
                            f"exceeds plausible antibody-domain range (>{max_len}aa)")
        return True, "ANARCI unavailable -- passed length heuristic only (weaker check)"

    try:
        anarci_res = anarci.anarci([("query", seq)], scheme="chothia", output=False)
    except Exception as e:
        return False, f"ANARCI raised an exception evaluating this sequence: {e}"

    numbered = bool(anarci_res and anarci_res[0] and anarci_res[0][0])
    if not numbered:
        return False, (f"ANARCI could not number this sequence as an Ig V-domain "
                        f"(length {len(seq)}aa) -- likely a non-antibody sequence "
                        f"(e.g. an antigen fragment mislabeled as an antibody chain)")

    if too_long:
        return False, (f"ANARCI found partial Ig numbering but sequence length "
                        f"{len(seq)}aa is implausible for a single VH/VL/Fab chain "
                        f"-- treating as suspicious rather than accepting")

    return True, "ANARCI confirmed Ig V-domain numbering"


def lead_extraction_node(state: MasterState):
    """
    Bridge node (previously missing entirely — referenced in a commented-out
    graph edge as "Antibody Extractor" / lead_extraction_universal_node, but
    that function was never actually defined anywhere in the codebase).

    ROOT CAUSE this fixes: the Biologist / Biologist Revisor store their
    results in the nested `found_antibodies` structure
    (found_antibodies -> molecular_design -> antibody_leads -> antibodies ->
    [{name, heavy_chain, light_chain, evidence_url}, ...]), using SINGULAR
    key names per-antibody. But every downstream consumer — Section 3 of
    the presenter, json_export_node, antibody_optimization_node, and
    mutation_application_node — reads the flat, PLURAL top-level state
    fields `antibody_leads` (list of names), `heavy_chains` (dict
    name->sequence), `light_chains` (dict name->sequence), and
    `ab_sequence_evidence_urls` (dict name->url). Nothing was ever
    translating between the two shapes, so those flat fields stayed empty
    on every single run — which is why Section 3 always showed "(0)"
    leads, why engineering always fell back to the hardcoded Trastuzumab
    Fab sequence regardless of what was actually retrieved, and why
    "discovered_leads" always exported as null in the JSON.
    """
    print("--- [NODE] Lead Extraction (bridging found_antibodies -> flat state fields) ---")

    found_abs = state.get("found_antibodies", {})
    antibody_records = normalize_antibody_list(found_abs)

    leads = []
    heavy_map = {}
    light_map = {}
    url_map = {}
    rejected_leads = []

    for rec in antibody_records:
        name = rec.get("name")
        if not name or not isinstance(name, str):
            continue

        # Per-antibody records use singular key names (schema in the
        # Biologist Revisor prompt: "heavy_chain"/"light_chain"), but be
        # defensive and also accept the plural forms in case a different
        # LLM response shape slips through.
        vh = rec.get("heavy_chain") or rec.get("heavy_chains")
        vl = rec.get("light_chain") or rec.get("light_chains")
        url = rec.get("evidence_url") or rec.get("ab_sequence_evidence_urls")

        # If this antibody is missing a sequence, try the deterministic
        # 4-tier cascade directly here — this is the earliest point in the
        # pipeline where we actually know a specific antibody name, so this
        # is the right place to attempt retrieval rather than leaving it to
        # evidence_consensus_node's much later, ADC-name-derived fallback.
        if not vh or not vl:
            print(f"   [LeadExtraction] Missing sequence(s) for '{name}', trying 4-tier cascade...")
            try:
                seq_result = antibody_sequence_search.invoke(
                    {"antibody_name": name, "target_antigen": state.get("target_antigen", "")}
                )
                if seq_result.get("status") == "Success":
                    vh = vh or seq_result.get("heavy_chains", {}).get(name)
                    vl = vl or seq_result.get("light_chains", {}).get(name)
                    url = url or seq_result.get("ab_sequence_evidence_urls")
            except Exception as e:
                print(f"   [LeadExtraction] Cascade fallback failed for '{name}': {e}")

        # VALIDATION GATE: reject either chain if it doesn't plausibly look
        # like an antibody variable domain (catches the antigen-as-antibody
        # fabrication failure mode BEFORE it's exported as a successful
        # lead). A rejected chain is nulled out here, converting what would
        # have been a silent "fab" outcome into an honest "null" outcome.
        if vh:
            ok, reason = _is_plausible_antibody_sequence(vh)
            if not ok:
                print(f"   [Validation] REJECTED heavy chain for '{name}': {reason}")
                rejected_leads.append({"name": name, "chain": "H", "reason": reason,
                                        "rejected_sequence_preview": vh[:60]})
                vh = None
        if vl:
            ok, reason = _is_plausible_antibody_sequence(vl)
            if not ok:
                print(f"   [Validation] REJECTED light chain for '{name}': {reason}")
                rejected_leads.append({"name": name, "chain": "L", "reason": reason,
                                        "rejected_sequence_preview": vl[:60]})
                vl = None

        leads.append(name)
        if vh:
            heavy_map[name] = vh
        if vl:
            light_map[name] = vl
        if url:
            url_map[name] = url

    print(f"   [LeadExtraction] Extracted {len(leads)} lead(s): {leads}")
    if rejected_leads:
        print(f"   [LeadExtraction] Validation gate rejected {len(rejected_leads)} "
              f"chain(s) as implausible antibody sequences: {rejected_leads}")

    # FIX: this is a real return (state update), unlike the earlier in-place
    # `state[...] = ...` mutations elsewhere in the codebase that never
    # actually persisted because LangGraph only tracks returned updates.
    return {
        "antibody_leads": leads,
        "heavy_chains": heavy_map,
        "light_chains": light_map,
        "ab_sequence_evidence_urls": url_map,
        "rejected_leads": rejected_leads,
    }


