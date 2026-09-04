"""
benchmark.py

The 9-prompt benchmark set used for the paper's Retrieval Precision (RP),
Task Completion Rate (TCR), and Audit Depth (AD) evaluation, plus the
run-mode switch (full benchmark vs. single prompt) and the mermaid graph
preview at the end.

NOTE: exactly like in the original single-file script, importing this
module EXECUTES the benchmark run (or single-prompt run) immediately --
this was module-level code in the original file, not wrapped in a
function, and that behavior is preserved unchanged here.

Split out of adcAgent_REVEAL_v2.py without changing any logic.
"""

from graph_build import phd_system


# --- RUNNING THE 9-PROMPT BENCHMARK SET ---
# These are the 9 prompts used for the paper's Retrieval Precision (RP),
# Task Completion Rate (TCR), and Audit Depth (AD) evaluation. #1-4 preserve
# the exact wording already present in prior versions of this script
# (Trastuzumab case study, colorectal/HER2, ovarian/FRa, myeloma/BCMA);
# #5-9 extend to the remaining antigen range the paper describes ("common
# targets (HER2) to complex biomarkers (Nectin-4)") and to the HGNC examples
# the paper itself names in Fig. 3 (ERBB2, CD274, MSLN).

BENCHMARK_PROMPTS = [

    # 1. HER2 (ERBB2) -- Trastuzumab case study, Fig. 7 in the paper
    # NOTE on ground truth: this prompt admits TWO clinically valid targets.
    # ERBB2 (T-DXd, DESTINY-Breast04) is the HER2-low-directed answer, but
    # TACSTD2/TROP2 (sacituzumab govitecan, ASCENT) is the guideline-standard
    # ADC for metastatic TNBC and is an equally defensible response to a
    # "bystander effect / low antigen density" brief. Benchmark scoring
    # accepts either; see Methods. The original wording said "HER-low",
    # which is not a real biomarker term -- corrected to "HER2-low".
    {"patient_description": (
        "54-year old female with Metastatic Breast Cancer. "
        "Biomarkers: HER2-low (IHC 1+), ER/PR negative (Triple Negative profile). "
        "Clinical Goal: Identify an ADC capable of utilizing the bystander effect "
        "to overcome low antigen density."
    )},

    # 2. HER2 (ERBB2) -- colorectal cancer
    {"patient_description": (
        "62-year-old male, diagnosed with Stage IV Metastatic Colorectal Cancer. "
        "Biomarkers: KRAS wild-type, high CEA levels, and HER2 overexpressed (IHC 3+). "
        "Co-morbidities: Mild cardiotoxicity from previous anthracycline exposure."
    )},

    # 3. FRa (FOLR1) -- ovarian cancer
    {"patient_description": (
        "54-year-old female with refractory ovarian cancer. "
        "Immunohistochemistry (IHC) reveals 3+ staining for Folate Receptor Alpha. "
        "Clinical Goal: Identify an ADC option after platinum-resistant relapse."
    )},

    # 4. BCMA (TNFRSF17) -- multiple myeloma
    {"patient_description": (
        "58-year-old female presents with refractory Multiple Myeloma. "
        "Bone marrow biopsy shows 80% plasma cells with high BCMA expression. "
        "Previous treatments included proteasome inhibitors with limited response."
    )},

    # 5. TROP2 (TACSTD2) -- triple-negative breast cancer (cf. ASCENT trial, ref [18])
    {"patient_description": (
        "49-year-old female with metastatic Triple-Negative Breast Cancer. "
        "Biomarkers: TROP2-positive by IHC, ER/PR/HER2 negative. "
        "Clinical Goal: Identify an ADC suitable after prior chemotherapy failure."
    )},

    # 6. Nectin-4 (NECTIN4) -- urothelial carcinoma (cf. EV-301 trial, ref [19])
    {"patient_description": (
        "67-year-old male with locally advanced/metastatic Urothelial Carcinoma. "
        "Biomarkers: Nectin-4 overexpression confirmed via IHC, prior platinum-based "
        "chemotherapy and checkpoint inhibitor failure. "
        "Clinical Goal: Identify a targeted ADC option for this refractory setting."
    )},

    # 7. CD30 -- Hodgkin lymphoma (cf. Brentuximab vedotin, refs [1],[6])
    {"patient_description": (
        "38-year-old male with relapsed/refractory classical Hodgkin Lymphoma. "
        "Biomarkers: CD30-positive Reed-Sternberg cells confirmed on biopsy. "
        "Previous treatment: ABVD chemotherapy with early relapse."
    )},

    # 8. CD33 -- acute myeloid leukemia (cf. Gemtuzumab ozogamicin, refs [1],[21]-[25])
    {"patient_description": (
        "71-year-old male with newly diagnosed CD33-positive Acute Myeloid Leukemia. "
        "Biomarkers: CD33 expression >80% by flow cytometry, intermediate cytogenetic risk. "
        "Co-morbidities: Reduced cardiac ejection fraction limiting induction chemotherapy options."
    )},

    # 9. MSLN (Mesothelin) -- matches the paper's own worked HGNC example, Fig. 3
    {"patient_description": (
        "60-year-old female with advanced Pancreatic Ductal Adenocarcinoma. "
        "Biomarkers: Mesothelin (MSLN) overexpression confirmed via IHC, "
        "microsatellite-stable disease. "
        "Clinical Goal: Identify an ADC candidate given limited response to "
        "standard gemcitabine-based chemotherapy."
    )},
]

# --- MODE SWITCH ---
# Set to True to run the full 9-prompt benchmark sequentially (recommended
# for regenerating RP / TCR / AD numbers for the paper revision).
# Set to False to run a single prompt (fast iteration while debugging).
RUN_FULL_BENCHMARK = True
SINGLE_TEST_INDEX = 0  # which BENCHMARK_PROMPTS entry to use when not running the full set

# CHANGE THIS NUMBER to control how many times each of the 9 prompts is run.
# Total runs in one execution = REPEATS_PER_PROMPT * 9.
# e.g. REPEATS_PER_PROMPT = 10  ->  90 total runs
#      REPEATS_PER_PROMPT = 20  ->  180 total runs
REPEATS_PER_PROMPT = 1

if RUN_FULL_BENCHMARK:
    all_results = []
    total_planned = len(BENCHMARK_PROMPTS) * REPEATS_PER_PROMPT
    run_counter = 0

    for idx, prompt in enumerate(BENCHMARK_PROMPTS):
        prompt_number = idx + 1

        for rep in range(REPEATS_PER_PROMPT):
            run_counter += 1
            rep_number = rep + 1
            print("\n" + "#" * 90)
            print(f"# BENCHMARK RUN {run_counter}/{total_planned}  "
                  f"(prompt {prompt_number}/{len(BENCHMARK_PROMPTS)}, repeat {rep_number}/{REPEATS_PER_PROMPT})")
            print("#" * 90)
            run_input = dict(prompt)
            run_input["benchmark_index"] = idx  # traced through to json_export_node's filename
            run_input["repeat_index"] = rep     # NEW: distinguishes repeats of the same prompt in filenames
            try:
                run_result = phd_system.invoke(run_input)
                all_results.append({
                    "prompt_index": prompt_number, "repeat": rep_number,
                    "status": "success", "result": run_result
                })
                print(f"DYNAMICALLY IDENTIFIED ANTIGEN: {run_result.get('target_antigen')}")
            except Exception as e:
                print(f" [!] Run {run_counter} (prompt {prompt_number}, repeat {rep_number}) failed: {e}")
                all_results.append({
                    "prompt_index": prompt_number, "repeat": rep_number,
                    "status": "failed", "error": str(e)
                })

    successes = sum(1 for r in all_results if r["status"] == "success")
    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPLETE: {successes}/{total_planned} runs succeeded "
          f"(Task Completion Rate = {successes/total_planned*100:.1f}%)")
    print("=" * 60)

    # Optional: per-prompt breakdown, useful for spotting which specific
    # prompt is driving failures rather than only seeing the pooled TCR.
    print("\nPer-prompt breakdown:")
    for idx in range(len(BENCHMARK_PROMPTS)):
        prompt_number = idx + 1
        prompt_runs = [r for r in all_results if r["prompt_index"] == prompt_number]
        prompt_successes = sum(1 for r in prompt_runs if r["status"] == "success")
        print(f"  Prompt {prompt_number}: {prompt_successes}/{len(prompt_runs)} succeeded")
else:
    test_input = dict(BENCHMARK_PROMPTS[SINGLE_TEST_INDEX])
    test_input["benchmark_index"] = SINGLE_TEST_INDEX
    result = phd_system.invoke(test_input)

    print(f"DYNAMICALLY IDENTIFIED ANTIGEN: {result['target_antigen']}")
    print("="*40)
    print(f"DYNAMICALLY IDENTIFIED SEQUENCE: {result['antigen_sequence']}")

from IPython.display import Image, display

try:
    display(Image(phd_system.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
