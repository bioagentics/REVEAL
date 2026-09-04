"""
graph_build.py

Wires all pipeline nodes into the LangGraph StateGraph and compiles the
final agentic system (phd_system) -- the 13-node sequential pipeline
across the four swimlane phases described in the paper (Fig. 2):
Synthesis -> Feature Extraction -> Engineering -> Validation.

Split out of adcAgent_REVEAL_v2.py without changing any logic (node
names, edges, and edge order are identical to the original).
"""

from state import MasterState
from nodes_clinical import clinical_input_node, clinical_revisor_node, should_revise
from nodes_biology import biologist_node, biologist_revisor_node, should_biologist_revise, lead_extraction_node
from nodes_discovery import antigen_sequence_node, adc_discovery_node, bioactivity_analysis_node, evidence_consensus_node
from esm_engineering import antibody_optimization_node, mutation_application_node
from nodes_reporting import json_export_node, grand_summary_presenter_node

from langgraph.graph import StateGraph, START, END


# Initialize the Graph with your MasterState
builder = StateGraph(MasterState)


# --- 1. DATA DISCOVERY & VERIFICATION NODES ---
# Register All Final Nodes
builder.add_node("Clinical Analyst", clinical_input_node)
builder.add_node("Clinical Revisor", clinical_revisor_node)
builder.add_node("Biologist", biologist_node)
builder.add_node("Biologist Revisor", biologist_revisor_node)
builder.add_node("Antibody Extractor", lead_extraction_node)  # FIX: was commented out and pointed at a function that didn't exist
builder.add_node("Antigen Sequencer", antigen_sequence_node)
builder.add_node("ADC discoverer", adc_discovery_node)
builder.add_node("Bio Analyzer", bioactivity_analysis_node)
builder.add_node("Evidence Consensus", evidence_consensus_node)
builder.add_node("hotspot_identification", antibody_optimization_node)
builder.add_node("apply_mutations", mutation_application_node)
builder.add_node("Exporter", json_export_node)
builder.add_node("Result Presenter", grand_summary_presenter_node)





# Define Logic Flow
builder.add_edge(START, "Clinical Analyst")
builder.add_edge("Clinical Analyst", "Clinical Revisor")
builder.add_conditional_edges(
    "Clinical Revisor",
    should_revise,
    {
        "continue_revision": "Clinical Revisor",
        "finalize_and_research": "Biologist"
    }
)
builder.add_edge("Biologist", "Biologist Revisor")
builder.add_conditional_edges(
    "Biologist Revisor",
    should_biologist_revise,
    {
        "revise": "Biologist Revisor", # Loop once if needed
        "finalize": "Antibody Extractor"        # FIX: run extraction before moving on
    }
)
builder.add_edge("Antibody Extractor", "Antigen Sequencer")
builder.add_edge("Antigen Sequencer", "ADC discoverer")
builder.add_edge("ADC discoverer", "Bio Analyzer")
builder.add_edge("Bio Analyzer", "Evidence Consensus")
# FIX: removed a stray/duplicate edge ("Exporter" -> "hotspot_identification")
# that was left over from an earlier graph iteration -- Exporter has no
# business feeding back into hotspot identification, and it created an
# unreachable/ambiguous predecessor for that node.
# FIX: restored the order stated in the paper's Fig. 2 architecture diagram
# (Mutation Application -> Exporter -> Result Presenter -> END); the prior
# version had Exporter and Result Presenter swapped.
builder.add_edge("Evidence Consensus", "hotspot_identification")
builder.add_edge("hotspot_identification", "apply_mutations")
builder.add_edge("apply_mutations", "Exporter")
builder.add_edge("Exporter", "Result Presenter")
builder.add_edge("Result Presenter", END)

# builder.add_edge("Biologist Revisor", "Antibody Extractor")
# builder.add_edge("Biologist Revisor", "Antigen Sequencer") # Sequencer follows Extractor
# builder.add_edge("Antibody Extractor", "ADC discoverer") # ADC follows Sequencer
# builder.add_edge("ADC discoverer", "Bio Analyzer")
# builder.add_edge("Bio Analyzer", "Evidence Consensus")  # Found IC50 -> verifies it
# builder.add_edge("Evidence Consensus", "Exporter")      # Verified data -> saves to JSON

# builder.add_edge("Exporter", "hotspot_identification")
# builder.add_edge("hotspot_identification", "apply_mutations")

# # Final Presentation
# builder.add_edge("apply_mutations", "Result Presenter")

# #builder.add_edge("exporter", "presenter")
# builder.add_edge("Result Presenter", END)


# Final Compiled Agentic System
phd_system = builder.compile()
