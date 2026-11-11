Run `hybrid_rerank-expand` | retrieval `hybrid_rerank` | k=5 | LLM `groq:openai/gpt-oss-120b` | 2026-11-11

| Metric | Result |
|---|---|
| Final answer correct / partial / incorrect (answerable questions) | 92% / 4% / 4% (n=26) |
| Correct before the verify loop (first draft) | 92% |
| False refusals on answerable questions | 0% |
| Correct refusals on unanswerable questions | 100% (n=4) |
| Golden page found in retrieved passages | 100% |
| Sentences flagged by the verifier, first draft -> final | 3.3% -> 0.0% |
| Answers with at least one flagged sentence, first draft -> final | 4% -> 0% |
| Mean trust score, first draft -> final | 0.96 -> 1.00 |
| Answers that needed a rewrite | 3% (average 0.03 rewrites per answer) |
| Planted lie detected / removed from the final answer | 100% / 100% (n=10) |
| Latency per question, average / p95 | 34.7s / 76.5s |
