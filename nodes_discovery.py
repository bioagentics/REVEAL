"""
nodes_discovery.py

Antigen sequence retrieval (UniProt), ADC record discovery, bioactivity
(IC50) enrichment, and cross-source evidence consensus.

Split out of adcAgent_REVEAL_v2.py without changing any logic.
"""

from state import MasterState
from config import llm, tavily_tool
from tools import openai_formatted_tools

import re
import requests
from langchain_core.messages import HumanMessage

def antigen_sequence_node(state: MasterState):
    raw_antigen = state.get("target_antigen")
    if not raw_antigen or raw_antigen == "Unknown":
        return {"antigen_sequence": None}

    print(f"--- Node: Antigen Sequencer (Normalizing: {raw_antigen}) ---")

    # 1. NORMALIZE: Convert Clinical Name to HGNC Gene Symbol
    # This prevents searching for "Her2" and getting wrong results; it finds "ERBB2"
    norm_prompt = (
        f"Identify the official HGNC Gene Symbol for the human protein commonly known as '{raw_antigen}'. "
        "Return ONLY the symbol (e.g., ERBB2, CD274, MSLN)."
    )
    norm_res = llm.invoke([HumanMessage(content=norm_prompt)])
    gene_symbol = norm_res.content.strip().upper()

    # Remove any stray text like "The symbol is..."
    gene_symbol = re.sub(r'[^A-Z0-9]', '', gene_symbol)
    print(f"   [Normalization] {raw_antigen} -> {gene_symbol}")

    # 2. SEARCH: Get the UniProt Accession using the Gene Symbol
    # We query for human (9606) and the specific gene symbol
    search_url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_symbol} AND organism_id:9606&format=json"

    try:
        search_response = requests.get(search_url)
        search_response.raise_for_status()
        search_data = search_response.json()

        if not search_data.get("results"):
            print(f"   [Error] No UniProt entry found for gene: {gene_symbol}")
            return {"antigen_sequence": None}

        # Get the first (canonical) result
        primary_entry = search_data["results"][0]
        uniprot_id = primary_entry["primaryAccession"]

        # 3. RETRIEVE: Get the FASTA sequence
        fasta_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        fasta_res = requests.get(fasta_url)
        fasta_res.raise_for_status()

        # Parse sequence
        lines = fasta_res.text.strip().split('\n')
        sequence = "".join(lines[1:])

        print(f"   [Success] Retrieved {len(sequence)} AA for {gene_symbol} ({uniprot_id})")

        return {
            "target_antigen": gene_symbol, # Update state with the official symbol
            "antigen_sequence": sequence,
            "antigen_evidence_url": f"https://www.uniprot.org/uniprotkb/{uniprot_id}/entry"
        }

    except Exception as e:
        print(f"   [Bio-API Error] {e}")
        return {"antigen_sequence": None}

import json
import re
#working
def adc_discovery_node(state: MasterState):
    antigen = state.get("target_antigen")
    print(f"--- Node: Chemical & ADC Discovery (Target: {antigen}) ---")

    # 1. STEP 1: Broad Search for ADC Landscape
    search_query = (
        f"Search for all Antibody-Drug Conjugates (ADCs) targeting {antigen}. "
        "Identify the ADC name, Payload name, Linker name, and reported IC50 values "
        "(in vitro or in vivo). Return as much detail as possible."
    )
    search_res = llm.bind_tools(openai_formatted_tools).invoke([HumanMessage(content=search_query)])
    raw_research = search_res.content

    # 2. STEP 2: Chemical SMILES Lookup
    # We ask the LLM to identify the chemical components found and then
    # use its internal knowledge or a tool to provide their SMILES strings.
    smiles_prompt = (
        "Based on the research above, identify the unique Payloads and Linkers. "
        "Provide the canonical SMILES string for each Payload and Linker mentioned. "
        "Ensure you distinguish between the Payload alone and the Linker-Payload complex."
        f"\n\nContext: {raw_research}"
    )
    smiles_res = llm.invoke(smiles_prompt)

    # 3. STEP 3: Integrated Parsing
    # We consolidate the research and chemical data into a final JSON structure
    final_extraction_prompt = (
        "Extract every distinct ADC from the text into a JSON list. "
        "For each ADC, include these fields: "
        "'adc_name', 'payload_name', 'payload_smiles', 'linker_name', 'linker_smiles', "
        "'status', 'ic50_value', 'binding_affinity_kd', 'indication'. "
        "If a specific SMILES or IC50 is not explicitly mentioned, provide the most "
        "common known value for that chemical name or return null."
        f"\n\nResearch Data: {raw_research}\n\nChemical Data: {smiles_res.content}"
    )

    json_output = llm.invoke(final_extraction_prompt)



    # Inside adc_discovery_node after llm.invoke...
    try:
        clean_json = re.sub(r'```json\n?|```', '', json_output.content).strip()
        adc_data = json.loads(clean_json)

        # Ensure we are returning a list, not a string or the 'adcs' wrapper dict
        if isinstance(adc_data, dict):
            final_list = adc_data.get("adcs") or adc_data.get("identified_adcs") or [adc_data]
        else:
            final_list = adc_data

    except Exception as e:
        print(f"   [Error] Could not parse JSON in Discovery Node: {e}")
        final_list = [] # Return empty list so the Output node doesn't crash



    print(f"   [Discovery] Captured {len(final_list)} benchmarks with chemical structures.")
    print("Final list", final_list)

    return {
        "found_adcs": final_list
    }

import json
import re

def bioactivity_analysis_node(state: MasterState):
    adcs = state.get("found_adcs", [])
    if not adcs or not isinstance(adcs, list):
        print("--- Node: Bioactivity Analysis (Skipped: No ADCs to analyze) ---")
        return {"found_adcs": adcs}

    print(f"--- Node: Bioactivity & Cheminformatics Analysis ({len(adcs)} leads) ---")

    enriched_results = []

    for adc in adcs:
        adc_name = adc.get("adc_name", "Unknown")
        payload = adc.get("payload_name", "Unknown")

        # 1. Targeted Bioactivity Query
        # We focus on capturing IC50 and PK/PD parameters
        bio_query = (
            f"Retrieve pharmacological data for {adc_name} (Payload: {payload}). "
            "Identify: IC50/EC50 values (in nM), Molecular Weight (MW) of the payload, "
            "LogP (hydrophobicity), and the exact stoichiometry (DAR) if available."
        )

        # Invoke the tool for deep scientific search
        search_res = tavily_tool.invoke(bio_query)

        # 2. Cheminformatics Normalization
        # We use the LLM as a 'Reasoning Engine' to standardize units
        refinement_prompt = (
            "Extract the following into a JSON object. Convert all concentration units to nanomolar (nM). "
            "If MW or LogP are not in the text, estimate them based on the chemical name. "
            "Keys: 'ic50_nm', 'molecular_weight_da', 'logp_value', 'dar_ratio', 'assay_cell_line'. "
            f"\n\nContext: {search_res}"
        )

        enrichment = llm.invoke(refinement_prompt)

        try:
            clean_json = re.sub(r'```json\n?|```', '', enrichment.content).strip()
            chem_data = json.loads(clean_json)
            # Merge the new bioinformatics data into the existing ADC dictionary
            adc.update(chem_data)
        except:
            print(f"   [Warning] Could not refine chemical data for {adc_name}")

        enriched_results.append(adc)

    return {
        "found_adcs": enriched_results
    }

import re

# NOT CALLED (revision cleanup, confirmed via grep -- no call sites anywhere
# in this file besides its own definition). The name-normalization job this
# was meant for is actually handled by _normalize_antibody_name() earlier in
# the file, which is the version that's actually wired into the retrieval
# cascade.
# def clean_antibody_name(full_name):
#     """
#     Extracts the base antibody name.
#     Example: 'Trastuzumab emtansine' -> 'Trastuzumab'
#     """
#     if not full_name or not isinstance(full_name, str):
#         return ""
#     # 1. Split by space and take the first word
#     print("Ful name", full_name)
#     base_name = full_name.split()[0]
#     print("Antibody extracted name:",base_name)
#
#     # 2. Remove common non-specific punctuation
#     base_name = re.sub(r'[,.;:]', '', base_name)
#
#     # 3. Validation: Ensure it ends with 'mab' (case insensitive)
#     if base_name.lower().endswith('mab'):
#         return base_name
#
#     return full_name # Fallback to original if 'mab' pattern isn't found

def evidence_consensus_node(state: MasterState):
    adcs = state.get("found_adcs", [])
    print(f"--- Node: Evidence Consensus & Pivot ({len(adcs)} entries) ---")
    verified_adcs = []

    for adc in adcs:
        name = adc.get("adc_name")

        # FIX: the old antibody-sequence fallback here was both broken
        # (mutated `state[...]` in place, which LangGraph never persists —
        # only returned dict updates are tracked) and redundant now that
        # lead_extraction_node runs earlier in the pipeline and handles
        # sequence retrieval directly against the Biologist's actual named
        # leads. Removed rather than duplicated/fixed here.

        current_ic50 = adc.get("ic50_nm")

        # PIVOT LOGIC: If ADCdb failed to provide IC50
        if not current_ic50 or current_ic50 == "N/A":
            print(f"   [Pivot] Searching for missing IC50 for {name}...")
            # Search for the payload potency as a fallback
            pivot_query = (
                f"Identify the reported IC50 value for {name} or its payload "
                f"{adc.get('payload_name')} in antigen-positive cell lines. "
                "Look specifically for values in nM or uM."
            )
            verification_raw = tavily_tool.invoke(pivot_query)
        else:
            # Verification of existing value
            check_query = f"Verify the IC50 value of {current_ic50} nM for {name}."
            verification_raw = tavily_tool.invoke(check_query)

        # AGENTIC REASONING: Standardize and Validate
        consensus_prompt = (
            "Review the research data and extract a confirmed IC50 value. "
            "If multiple values exist, provide the one from the most recent study. "
            "Return JSON: {'verified_ic50': float, 'unit': 'nM', 'confidence': 'High/Med/Low'}"
            f"\n\nData: {verification_raw}"
        )

        res = llm.invoke(consensus_prompt)

        # FIX: this is where the function used to just stop — the consensus
        # result was computed and thrown away, and `verified_adcs` was never
        # appended to or returned, so nothing from this node ever reached
        # later state. Now we actually parse and merge it.
        try:
            clean_json = re.sub(r'```json\n?|```', '', res.content).strip()
            consensus_data = json.loads(clean_json)
            adc["verified_ic50"] = consensus_data.get("verified_ic50")
            adc["verified_ic50_unit"] = consensus_data.get("unit")
            adc["verified_ic50_confidence"] = consensus_data.get("confidence")
        except Exception as e:
            print(f"   [Warning] Could not parse consensus IC50 for {name}: {e}")

        verified_adcs.append(adc)

    # FIX: function previously had no return statement at all (implicit
    # None), so LangGraph treated this node as a complete no-op regardless
    # of what it computed.
    return {
        "found_adcs": verified_adcs
    }

