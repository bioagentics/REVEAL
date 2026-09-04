"""
nodes_clinical.py

Clinical analysis phase: initial Dr. M analyst node, the reflect/revise
loop (clinical_revisor_node), and the should_revise routing function.

Split out of adcAgent_REVEAL_v2.py without changing any logic.
"""

from state import MasterState, AnswerQuestion, ReviseAnswer
from config import llm, tavily_tool, revise_instructions

"""Clinical input node

"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

# 1. Define the System Message with the variable explicitly
system_message_content = """You are Dr. M, a Principal Investigator in Molecular Oncology and Cancer Immunology. Your expertise lies in identifying tumor-associated antigens (TAAs) and designing high-affinity monoclonal antibodies for targeted therapy.

        Your response must adhere to this rigorous academic framework:
        1. {first_instruction}
        2. **Antigen Profile:** Identify a specific cancer type and a target antigen (e.g., HER2, CD20, EGFR, or a neoantigen). Explain the antigen's expression profile, focusing on differential expression between malignant and healthy tissues.
        3. **Antibody Mechanism:** Detail the corresponding antibody's mechanism of action—whether it involves Antibody-Dependent Cellular Cytotoxicity (ADCC), Complement-Dependent Cytotoxicity (CDC), or signaling pathway inhibition.
        4. **Clinical Challenges:** Discuss current hurdles such as "off-tumor" toxicity, antigen escape, or the immunosuppressive tumor microenvironment (TME).
        5. **Supervisor Reflection:** Critically evaluate your technical depth. Ensure you have addressed the biochemical binding affinity (Kd values) or the structural biology of the paratope-epitope interface.
        6. **Research Synthesis:** Provide 1-3 specific search queries for PubMed or Google Scholar to further investigate the latest Phase II/III clinical trial data or structural proteomics related to this target.

        Tone: Professional, analytical, and intellectually demanding. Use precise terminology (e.g., glycosylation patterns, isotype selection, intracellular signaling cascades) and also breifly explain in case someone is not familiar with it.
"""
# 2. Create the template using the proper constructor
prompt_template = ChatPromptTemplate(
    messages=[
        ("system", system_message_content),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Synthesize the response as Dr. M, ensuring the transition is seamless.")
    ],
    input_variables=["first_instruction", "messages"]
)

# 3. Now your partials will work correctly
#first_responder_prompt = prompt_template.partial(first_instruction="Provide a detailed ~250 word answer")

import re

def extract_references_from_text(text: str) -> list:
    # 1. Check if the text is even arriving
    if not text:
        print("ERROR: Input text is empty or None.")
        return []

    # 2. Improved Regex:
    # Matches 'References' with any number of '#' or '*' around it,
    # and allows for optional whitespace/colon at the end.
    pattern = r'(?i)[\*#\s]*references[\*#\s]*:?'

    # Find all matches to see if the header exists at all
    matches = re.findall(pattern, text)
    # 3. Splitting the text
    parts = re.split(pattern, text)
    if len(parts) > 1:
        # We take the last part (the bibliography)
        reference_block = parts[-1].strip()


        # 4. Cleaning the lines
        lines = [line.strip() for line in reference_block.split('\n') if line.strip()]

        # 5. Filter for lines that start with [#] or #.
        # This prevents 'Search queries' or 'Summary' from being included
        final_refs = [l for l in lines if re.match(r'^\[?\d+\]?[\s\.]', l)]

        return final_refs

    print("CRITICAL: The split failed. The 'References' header was not recognized.")
    return []

def clinical_input_node(state: MasterState):
    print("--- Node: Dr. M Initial Analyst ---")

    # Setup the prompt with the specific instruction
    prompt = prompt_template.partial(first_instruction="Provide a detailed ~250 word answer")
    chain = prompt | llm.bind_tools(tools=[AnswerQuestion])

    # Invoke using the patient description
    response = chain.invoke({"messages": [HumanMessage(content=state["patient_description"])]})

    # Extract the answer from tool_calls or content
    if response.tool_calls:
        ans = response.tool_calls[0]["args"].get("answer", "")
    else:
        ans = response.content

    #ref_pattern = r'(?i)\*\*?references\*\*?:?'
    #extracted_refs = re.findall(ref_pattern, ans)


    new_refs = extract_references_from_text(ans)

    # Update state: Combine existing refs with new ones and deduplicate
    current_refs = state.get("references", [])
    updated_refs = list(dict.fromkeys(current_refs + new_refs))
    print("Dr. M Initial Analyses: ", ans)
    print("Updates refs clinical_input_node ", updated_refs)


    return {
        "initial_analysis": ans,
        "messages": [response],
        "iteration_count": 0, # Explicitly start the counter here
        "references": updated_refs
    }

import json
from langchain_core.messages import ToolMessage, HumanMessage

def clinical_revisor_node(state: MasterState):


    # Use .get() to prevent the KeyError we saw earlier
    current_iter = state.get("iteration_count")
    print(f"--- Node: Clinical Revisor (Pass {current_iter + 1}) ---")


    # 1. Identify the specific AI message that triggered the tool calls
    # We look back through the messages to find the one the API is complaining about
    last_ai_msg = state["messages"][-1]
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            last_ai_msg = msg
            break

    tool_messages = []
    if last_ai_msg:
        # We MUST provide a response for EVERY tool_call_id
        for tc in last_ai_msg.tool_calls:
            print(f"   [System] Resolving tool call ID: {tc['id']}")

            # Execute the actual search
            queries = tc["args"].get("search_queries", [])
            if queries:
                results = {q: tavily_tool.invoke(q) for q in queries}
            else:
                results = {"info": "No specific queries were suggested; proceeding with general revision."}

            # Create the mandatory ToolMessage
            tool_messages.append(ToolMessage(
                content=json.dumps(results),
                tool_call_id=tc["id"] # This matches the ID in the error
            ))

    # 2. Re-bind the Revisor Chain
    revisor_prompt = prompt_template.partial(first_instruction=revise_instructions)
    revisor_chain = revisor_prompt | llm.bind_tools(tools=[ReviseAnswer])

    # 3. Construct a VALID message history
    # Sequence: [Human Question] + [Analyst AI Message] + [Tool Messages]
    history = [HumanMessage(content=state["patient_description"])] + state.get("messages", []) + tool_messages

    # 4. Invoke the Revisor
    res_revised = revisor_chain.invoke({"messages": history})

    # 5. Extract data (Handling potential tool_call vs prose)
    if res_revised.tool_calls:
        revised_args = res_revised.tool_calls[0]["args"]
        final_report = revised_args.get("answer", "")
    else:
        final_report = res_revised.content
        # NOT USED (revision cleanup): this local variable is never read --
        # references actually come from extract_references_from_text(final_report)
        # a few lines below, applied uniformly whether or not tool_calls fired.
        # references = []


    extraction_prompt = (
        "You are a bioinformatics assistant. Extract the primary target antigen"
        "symbol from the text below. Return ONLY the symbol."
        f"\n\nText: {final_report}"
    )
    extraction_res = llm.invoke(extraction_prompt)
    target = extraction_res.content.strip()

    # 5. EXTRACTION LOGIC: Append new references
    #ref_pattern = r'(?i)\*\*?references\*\*?:?'
    #new_refs = re.findall(ref_pattern, final_report)

    new_refs = extract_references_from_text(final_report)

    # Update state: Combine existing refs with new ones and deduplicate
    current_refs = state.get("references", [])
    updated_refs = list(dict.fromkeys(current_refs + new_refs))

    print(f"--- Node: Clinical Revises Response (Pass {current_iter + 1}) ---", final_report)
    print("Updated refs clinical_revisor_node", updated_refs)


    return {
        "clinical_analysis": final_report,
        "target_antigen": target,
        "iteration_count": current_iter + 1,
        "references": updated_refs,
        # IMPORTANT: We add both the tool messages and the new AI message to state
        "messages": tool_messages + [res_revised]
    }

def should_revise(state: MasterState) -> str:
  # Rule 1: Use .get() to provide a fallback of 0 if the key is missing
    current_iter = state.get("iteration_count")
    print(f"Current Iteration: {current_iter}")

    if current_iter == 0:
        return "continue_revision"

    if current_iter >= 3:
        return "finalize_and_research"

    last_msg = state["messages"][-1]
    content = last_msg.content.lower()

    if "needs further investigation" in content or "insufficient data" in content:
        return "continue_revision"

    return "finalize_and_research"

