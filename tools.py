"""
tools.py

LangChain tool definitions used by the LangGraph agent nodes (web search,
UniProt lookups, ADC DB search) plus the tools registry consumed by
graph_build.py / the node modules.

Split out of adcAgent_REVEAL_v2.py without changing any logic.
"""

from typing import List, Dict, Any
import requests
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function

from config import tavily_tool
from sequence_lookup import antibody_sequence_search

@tool
def web_search(query: str) -> List[Dict[str, Any]]:
    """Search the web via Tavily and return results."""
    return tavily_tool.invoke({"query": query})

@tool
def uniprot_fasta(accession: str) -> str:
    """Fetch UniProt FASTA by accession."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text.strip()

import requests
from typing import Dict, Any, List

@tool
def adc_db_search(query: str, search_type: str = "antigen") -> Dict[str, Any]:
    """
    Search specialized databases for ADC clinical data.
    search_type: 'antigen' or 'antibody'
    """
    print(f"--- [Tool] Querying ADC Database for {query} ({search_type}) ---")

    # 1. Targeted Database Endpoint (Example: Public ADC data via API or Bio-Search)
    # For your PhD, you could use an API like PubChem for the chemical parts
    # or a Tavily-powered deep search limited to clinicaltrial.gov and ADC databases.

    search_instructions = (
        f"Search clinical databases for ADCs targeting {query}. "
        "Locate: ADC Name, Payload name, Linker name, Drug Status, and IC50. "
        "Crucially, find the SMILES strings for the linker and payload."
    )

    # Using the search tool internally to simulate the DB lookup
    search_results = tavily_tool.invoke(search_instructions)

    # 2. Structure the hit to be 'Parser-Friendly'
    # We return a dictionary that your nodes can easily iterate over
    return {
        "query": query,
        "type": search_type,
        "raw_results": search_results,
        "database_source": "ADC-Mining-V2"
    }

# CELL 8 — Tool: UniProt search (single responsibility)
import requests
@tool

def uniprot_search(query: str) -> Dict[str, Any]:
    """
    Search UniProt for a human protein by query (antigen string).
    Returns top results (json) with key fields.
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"({query}) AND (organism_id:9606)",
        "format": "json",
        "size": 5,
        "fields": "accession,protein_name,gene_primary,organism_name,reviewed",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


# CELL 11 — Tools registry (single responsibility)
tools = [web_search, uniprot_search, uniprot_fasta, antibody_sequence_search, adc_db_search]
tools_by_name = {t.name: t for t in tools}
# Convert each StructuredTool in your list into a dictionary the model understands
openai_formatted_tools = [convert_to_openai_function(t) for t in tools]
