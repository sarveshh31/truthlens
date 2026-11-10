Run `smoke` | retrieval `hybrid_rerank` | k=5 | LLM `groq:openai/gpt-oss-120b` | 2026-11-10

| Metric | Result |
|---|---|
| Final answer correct / partial / incorrect (answerable questions) | 67% / 33% / 0% (n=3) |
| Correct before the verify loop (first draft) | 67% |
| False refusals on answerable questions | 0% |
| Correct refusals on unanswerable questions | n/a (n=0) |
| Golden page found in retrieved passages | 100% |
| Sentences flagged by the verifier, first draft -> final | 0.0% -> 0.0% |
| Answers with at least one flagged sentence, first draft -> final | 0% -> 0% |
| Mean trust score, first draft -> final | 1.00 -> 1.00 |
| Answers that needed a rewrite | 0% (average 0.00 rewrites per answer) |
| Planted lie detected / removed from the final answer | n/a / n/a (n=0) |
| Latency per question, average / p95 | 9.5s / 16.1s |
