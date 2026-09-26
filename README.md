# AgentMT

Code and data for **"Say It Differently, Act Differently? Metamorphic Testing of Operational Behavior Consistency in AI Agents"**.

AgentMT tests whether an AI agent *acts* the same when a user expresses the same request differently. Each source task is rewritten into intent-preserving variants (tone, language, formulation); every source and variant is executed three times, and the resulting executions are compared at three levels: the **operation graph** (what the agent did), the **final output** (what it reported), and the **consequence** (the permissions it requested, the information it shared, or the tools it invoked).

## Repository Structure

```
AgentMT/
├── README.md
├── run_all.sh                         # rebuild every table and figure (no API calls)
├── requirements.txt
├── prompts/                           # every LLM prompt used (see prompts/README.md)
│
├── dataset/
│   ├── environments/                  # 116 source tasks + fixed, replayable environments
│   │   ├── fixed_sources.json         #   AIP (40) / AgentCIBench (36) / TRAJECT-Bench (40) source tasks
│   │   └── agentci_runtimes/          #   seeded AgentCIBench app states (one per scenario)
│   ├── source_selection/              # sampled source tasks and the 15 tone templates (forms.json)
│   ├── tone_taxonomy/                 # WildChat-derived tone template inventory
│   ├── variants/                      # intent-preserving variants of every source task
│   │   ├── tone/<task_id>/suite_ready_v3.json           # 15 tone variants (5 polite, 5 neutral, 5 impolite)
│   │   ├── multilingual/<task_id>/suite.json            # 7 languages (ar, de, es, fr, pt, ru, zh)
│   │   ├── formulation_reordering/                      # sentence reordering (preferred formulation variant)
│   │   ├── formulation_paraphrase/                      # GPT-4o locked-span paraphrase (fallback)
│   │   ├── locked_spans_ner/                            # entities locked during paraphrasing
│   │   └── information_order/
│   ├── traject_response_library/      # frozen tool responses that make TRAJECT-Bench deterministic
│   ├── subsets/                       # RQ3 AgentCIBench subset, mitigation subset
│   ├── mitigation_extraction/         # GPT-4o normalized English task specifications (270)
│   ├── terminal_bench/                # 15 selected Terminal-Bench 2.0 tasks and their variants
│   └── licenses/                      # licenses of the upstream benchmarks
│
├── runs/                              # agent executions, packed; restored by scripts/unpack_runs.py
│   ├── main/{gemini-2.5-flash, gpt-4.1-mini, claude-haiku-4.5}.jsonl.gz
│   ├── rq3_acb/{gpt-6-astra, clarify_*}.jsonl.gz
│   ├── mitigation/{gemini-2.5-flash, gpt-4.1-mini, claude-haiku-4.5}.jsonl.gz
│   └── terminal_bench/<harbor job>.jsonl.gz
│
├── results/                           # LLM-judge scores (shipped) + everything run_all.sh regenerates
│   ├── rq1_graph/  rq1_output/  rq2_consequence/  rq3_acb/  rq3_coding/  mitigation/
│
├── privacy_mt/                        # core library: environments, agent runners, operation graphs, WL similarity
└── scripts/
    ├── 01_dataset_construction/       # variant generation, subset selection, mitigation extraction
    ├── 02_agent_execution/            # main study, RQ3 (clarification / GPT-6 Astra), Terminal-Bench (Harbor)
    ├── 03_llm_judging/                # GPT-4o output similarity, AgentCIBench disclosure judge
    ├── 04_analysis/                   # RQ1-RQ3 and mitigation metrics
    ├── 05_tables_figures/             # all paper tables (.tex) and figures (.pdf)
    └── unpack_runs.py
```

## Dataset

### Source tasks

116 source tasks from three agent benchmarks, each executed in a fixed, replayable environment:

| Benchmark | Tasks | Agent interface | Consequence compared (RQ2) |
|---|---|---|---|
| AI Agent Permissions (AIP) | 40 | permission requests over personal data fields | set of requested (data field, scope) |
| AgentCIBench (ACB) | 36 | seeded apps (messenger, calendar, maps, shop, ...) | output channels + disclosed information items |
| TRAJECT-Bench (TJB) | 40 | real-world tools with frozen responses | multiset of invoked tools |

### Intent-preserving variants

| Family | Variants per task | Construction |
|---|---|---|
| Tone | 15 (`p01–p05` polite, `n01–n05` neutral, `i01–i05` impolite) | most frequent WildChat request templates of each tone, filled with the source task |
| Language | 7 (`lang_ar/de/es/fr/pt/ru/zh`) | direct translation with literal protection, audited in a separate context |
| Formulation | 1 | lossless sentence reordering; GPT-4o locked-span paraphrase when reordering is not possible |

Every variant was checked for task, authorization and information-order equivalence (`review` fields in the suite files), and protected literals (names, numbers, paths, quotes) were checked deterministically. In total: 116 tasks × (1 source + 23 variants) × 3 runs per model.

### Models

| Setting | Models |
|---|---|
| Main study (RQ1, RQ2) | Gemini 2.5 Flash (Vertex AI), GPT-4.1 Mini (`gpt-4.1-mini-2025-04-14`), Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) |
| RQ3: clarification | the three models above; GPT-4o (`gpt-4o-2024-11-20`) simulates the user, at most three answers |
| RQ3: frontier model | GPT-6 Astra |
| RQ3: coding agents | Claude Code + Claude Haiku 4.5, Codex CLI + GPT-6 Luna (Terminal-Bench 2.0 via Harbor) |
| Judges | GPT-4o (`gpt-4o-2024-11-20`, temperature 0) for output similarity, ACB disclosure and mitigation extraction |

The RQ3 and mitigation settings use 8 variants per task (`p01`, `n01`, `i01`, `lang_zh`, `lang_ru`, `lang_fr`, `lang_es`, `formulation`) on hash-sampled subsets (`dataset/subsets/`): 15 ACB tasks for RQ3, 10 tasks per benchmark for mitigation, and 15 Terminal-Bench 2.0 tasks for coding agents.

## Metrics

All three metrics compare a source prompt with its variant over the 3 × 3 source-run × variant-run pairs; the baseline (S–S) uses the 3 within-source pairs, so that run-to-run nondeterminism is separated from the effect of the variant. Scores are aggregated variant → family (within task) → task; Δ = S–V − S–S with a task-level bootstrap 95% CI.

| Metric | Definition | Script |
|---|---|---|
| Graph similarity | normalized Weisfeiler–Lehman kernel (h = 2, ordered edges) between operation graphs | `04_analysis/compute_rq1_fullgraph_3x3.py` |
| Output similarity | GPT-4o rubric score 0–4 / 4 between final responses (two empty responses = identical) | `03_llm_judging/evaluate_output_similarity_gpt4o.py`, `04_analysis/analyze_output_similarity.py` |
| Consequence inconsistency | fraction of pairs whose consequence (table above) differs | `04_analysis/analyze_rq2_consistency.py` |

## Run format

After `python3 scripts/unpack_runs.py`, every execution is a directory `runs/<setting>/<model>/<task_id>/<family>__<form>/r<k>/` with

- `events.jsonl`: tool calls, tool results, permission requests/grants and disclosures, in order;
- `operation_graph.json`: the operation graph used for graph similarity;
- `result.json`: status, final response, number of tool calls, token usage.

Each model directory also holds `manifest.json` (all executed prompts) and `run_summary.jsonl`. Terminal-Bench trials keep Harbor's `config.json`, `result.json` (verifier reward, exceptions) and the ATIF `agent/trajectory.json`.

## Reproducing the Results

### Tables and figures (no API calls)

```bash
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` unpacks the runs on first use, recomputes all metrics from the executions and the shipped GPT-4o scores, and writes every paper table (`.tex`) and figure (`.pdf`) under `results/`:

| Paper item | Output |
|---|---|
| Table 3 (graph similarity) | `results/rq1_graph/table3_rows.tex` |
| Figure 2 (graph similarity distributions) | `results/rq1_graph/figures/` |
| Figure 3 (example operation graphs) | `results/rq1_graph/figures/fig_example_graph_{source,variant}.pdf` |
| Output similarity and output-vs-graph scatter | `results/rq1_output/analysis/` |
| RQ2 table and figure | `results/rq2_consequence/table_rq2_consistency.tex`, `fig_rq2_consistency.pdf` |
| RQ3 clarification table, GPT-6 Astra figure | `results/rq3_acb/table_rq3_clarify.tex`, `fig_rq3_gpt6.pdf` |
| RQ3 coding-agent table | `results/rq3_coding/table_tb2.tex` |
| Mitigation table | `results/mitigation/table_mitigation.tex` |

### Re-running experiments (API keys required)

Put keys in `secrets/openai_key.txt` and `secrets/anthropic_key.txt` (gitignored); Gemini uses Vertex AI application-default credentials (pass `--project YOUR_GCP_PROJECT`).

```bash
# main study: 116 tasks x 24 forms x 3 runs
python3 scripts/02_agent_execution/run_full_graph_experiment.py --provider anthropic --model claude-haiku-4-5-20251001 \
    --output runs/main/claude-haiku-4.5 --api-key-file secrets/anthropic_key.txt

# GPT-4o output similarity
python3 scripts/03_llm_judging/evaluate_output_similarity_gpt4o.py --api-key-file secrets/openai_key.txt

# RQ3: GPT-6 Astra, and clarification (up to three questions, GPT-4o user simulator)
python3 scripts/02_agent_execution/run_rq3_acb.py --condition single --provider openai --model gpt-6-astra \
    --output runs/rq3_acb/gpt-6-astra --api-key-file secrets/openai_key.txt
python3 scripts/02_agent_execution/run_rq3_acb.py --provider anthropic --model claude-haiku-4-5-20251001 --condition clarify \
    --output runs/rq3_acb/clarify_claude-haiku-4.5 --api-key-file secrets/anthropic_key.txt

# mitigation: GPT-4o extraction, then the agents run on the extracted specifications
python3 scripts/01_dataset_construction/extract_mitigation_prompts.py --api-key-file secrets/openai_key.txt

# RQ3 coding agents: clone Terminal-Bench 2.0 (commit 2fd12b8) into dataset/terminal_bench/repo, then
python3 scripts/01_dataset_construction/prepare_tb2_variants.py --materialize-only
bash scripts/02_agent_execution/run_tb2_harbor.sh full
python3 scripts/02_agent_execution/retry_tb2_infra.py --job <job> --agent <agent> --model <model>   # infrastructure failures only
```

Every runner prints its options with `--help`; runners are resumable and refuse to write into an output directory created with a different configuration.

## Dependencies

- Python 3.11+, `numpy`, `matplotlib` (analysis); `google-genai`, `certifi` (agents and judges)
- `nltk`, `torch`, `transformers` (formulation variant construction only)
- [Harbor](https://github.com/laude-institute/harbor) and Docker (Terminal-Bench 2.0 only)

## Licenses of Upstream Data

AI Agent Permissions (CC BY 4.0), TRAJECT-Bench (MIT) and Terminal-Bench 2.0 (Apache-2.0); see `dataset/licenses/`. The AgentCIBench environments are derived from the AgentCIBench seed scenarios.
