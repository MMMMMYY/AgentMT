# GPT-4o constrained paraphrasing run

Date: 2026-09-18

## Configuration

- Model: `gpt-4o-2024-11-20`
- Temperature: `0`
- Maximum candidates per task: `2`
- Output: strict JSON schema
- Candidate generator only: GPT-4o is not the semantic-equivalence oracle.
- Locked spans: CoreNLP named entities plus application/tool names, identifiers,
  amounts, dates, time expressions, locations, URLs, channels, quoted strings,
  temporal ranges, and scope markers.

## Results

| Dataset | Paraphrasing tasks | At least one surface-valid candidate |
|---|---:|---:|
| AI Agent Permissions | 33 | 29 |
| AgentCIBench | 28 | 25 |
| TRAJECT-Bench | 12 | 3 |
| **Total** | **73** | **57** |

Together with the 43 retained lossless sentence-reordering variants, the
current deterministic surface coverage is **100/116** source tasks.

The generation and corrected retry used 92 API calls, 31,104 input tokens, and
16,536 output tokens. These counts include the 19-task diagnostic retry.

## Interpretation

GPT-4o substantially outperformed the QCPG pilot on ordinary short agent
requests and AgentCIBench scenarios. The remaining automatic rejections are
concentrated in long TRAJECT-Bench workflows and several very short prompts.
Manual inspection indicates that some rejected candidates are semantically
plausible but fail deliberately conservative surface rules, especially the
surface-similarity and length-ratio guards. They must be sent to independent
semantic and operational-equivalence review rather than accepted or discarded
solely from these surface scores.

Surface validity does not establish semantic equivalence. Formal acceptance
still requires unchanged action, object, recipient, destination, scope,
arguments, temporal constraints, modality, negation, and operation order.

## Artifacts

- `candidates.json`: selected initial/retry candidate sets for all 73 routed tasks.
- `summary.json`: merged aggregate statistics and unresolved task IDs.
- `../gpt4o_constrained_paraphrase_73_20260918/`: full initial run and raw responses.
- `../gpt4o_constrained_paraphrase_retry19_20260918/`: corrected 19-task retry.
