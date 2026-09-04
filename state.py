"""
state.py

Shared LangGraph state schema (MasterState) and the reflection/answer
Pydantic models used by the clinical revisor node.

Split out of adcAgent_REVEAL_v2.py without changing any logic.
"""

import operator
from typing import TypedDict, Annotated, List, Dict, Any, Union
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

class MasterState(TypedDict):

    patient_description: Annotated[str, lambda old, new: new]
    target_antigen: Annotated[str, lambda old, new: new]
    benchmark_index: Annotated[int, lambda old, new: new]  # which of the 9 prompts this run came from (0 if single-run mode)
    repeat_index: Annotated[int, lambda old, new: new]  # which repeat of that prompt this run is (0 if single-run mode)

    initial_analysis: str
    clinical_analysis: str
    antigen_sequence: str  # New field for the protein sequence (FASTA/String)
    antigen_evidence_url: str

    references: List[str] # Saving citations explicitly

    found_antibodies: Dict[str, Any] # Full JSON from Biologist
    antibody_leads: List[str]


    messages: Annotated[List[BaseMessage], operator.add]

    iteration_count: int            # For Clinical Revisor
    biologist_iterations: int        # For Biologist Revisor

    heavy_chains: Annotated[Dict[str, str], lambda old, new: {**old, **new}] # Format: {"Trastuzumab": "EVQLVE..."}
    light_chains: Annotated[Dict[str, str], lambda old, new: {**old, **new}] # Format: {"Trastuzumab": "DIQMTQ..."}
    ab_sequence_evidence_urls: Dict[str, str]  # Format: {"Trastuzumab": "https://..."}

    found_adcs: Dict[str, Any]


    hotspot_reports: dict

    # FIX (found via actual run output — the ROOT CAUSE of Section 6/7 always
    # being empty despite hotspots being correctly identified): this field was
    # never declared in the MasterState schema. LangGraph only persists state
    # keys that are part of the TypedDict schema; mutation_application_node
    # was correctly computing optimized_sequences and returning it every run,
    # but LangGraph silently dropped the key because it wasn't a recognized
    # channel, so grand_summary_presenter_node and json_export_node always
    # read back None/empty. hotspot_reports worked because it WAS declared —
    # that's why hotspots populated correctly but engineered sequences never did.
    optimized_sequences: List[Dict[str, Any]]

    # NEW (validation gate, added alongside Trastuzumab-fallback cleanup):
    # tracks any antibody chain that was retrieved but rejected by
    # _is_plausible_antibody_sequence() before being accepted as a lead --
    # e.g. a fabricated antigen-as-antibody substitution. Declared here for
    # the same reason optimized_sequences had to be declared above: LangGraph
    # only persists state keys that are part of this schema.
    rejected_leads: List[Dict[str, Any]]


class Reflection(BaseModel):
	missing: str = Field(description="What information is missing")
	superfluous: str = Field(description="What information is unnecessary")

class AnswerQuestion(BaseModel):
	answer: str = Field(description="Main response to the question")
	reflection: Reflection = Field(description="Self-critique of the answer")
	search_queries: List[str] = Field(description="Queries for additional research")

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""
    references: List[str] = Field(description="Citations motivating your updated answer.")
