# Prompts

All LLM prompts used in the study. Files other than `mitigation_extraction_system.txt` are exported verbatim from the code (the header line gives the source location); the code remains the source of truth. `{...}` marks values inserted at run time.

| File | Used for | Model |
|---|---|---|
| `agent_system_aip.txt` | agent system prompt, AI Agent Permissions | agents under test |
| `agent_system_acb.txt` | agent system prompt, AgentCIBench (clarification adds a sentence allowing up to three questions) | agents under test |
| `agent_system_tjb.txt` | agent system prompt, TRAJECT-Bench | agents under test |
| `variant_tone.txt` | split a source request into context + task before applying the tone templates | Gemini 2.5 Flash |
| `variant_translation.txt`, `variant_translation_audit.txt` | translation and separate-context equivalence audit | Gemini 2.5 Flash |
| `variant_formulation_paraphrase.txt` | locked-span constrained paraphrase (formulation fallback) | GPT-4o |
| `clarification_user_simulator.txt` | user simulator for the clarification setting (RQ3) | GPT-4o |
| `judge_output_similarity.txt` | final-response similarity rubric (0–4) | GPT-4o |
| `judge_acb_disclosure.txt` | which AgentCIBench information items were disclosed | GPT-4o |
| `mitigation_extraction_system.txt` | mitigation: normalized English task specification | GPT-4o |

The 15 tone templates are in `dataset/source_selection/forms.json`.
