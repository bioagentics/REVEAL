"""
main.py

Entrypoint for the REVEAL pipeline. This just imports the compiled
LangGraph system and then runs the benchmark module -- exactly the same
top-to-bottom execution order as the original single-file
adcAgent_REVEAL_v2.py, just spread across files:

    config.py            environment / API client setup (LLM, Tavily)
    sequence_lookup.py    4-tier antibody VH/VL sequence retrieval cascade
    tools.py              LangChain tool definitions + tools registry
    state.py               MasterState schema + reflection/answer models
    nodes_clinical.py      Clinical Analyst / Clinical Revisor nodes
    nodes_biology.py       Biologist / Biologist Revisor / lead extraction
    nodes_discovery.py     Antigen sequencer / ADC discovery / bioactivity /
                            evidence consensus
    esm_engineering.py     ESM-1v hotspot scan + mutation application
    nodes_reporting.py     Grand summary printout + JSON export
    graph_build.py         Wires all nodes into StateGraph -> phd_system
    benchmark.py            The 9-prompt benchmark set + run loop (runs on
                            import, same as the original script)

Run with:
    python main.py
"""

# Importing benchmark pulls in graph_build (which builds phd_system) and
# then executes the benchmark run / single-prompt run exactly as the
# original script did at module level.
import benchmark  # noqa: F401
