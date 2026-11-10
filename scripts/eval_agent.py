import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from app import config
from app.agenteval import (
    count_flags,
    grade,
    is_daily_limit,
    is_refusal,
    lie_detected,
    lie_removed,
    parse_wait,
    render_markdown,
    summarize,
)
from app.evalutils import first_relevant_rank, load_golden
from app.graph import MAX_REWRITES, run
from app.retrieval import MODES


class QuotaExhausted(Exception):
    pass


parser = argparse.ArgumentParser(description="End-to-end evaluation of the TruthLens agent")
parser.add_argument("--golden", default="eval/golden.json")
parser.add_argument("--mode", choices=MODES, default=config.RETRIEVAL_MODE, help="retrieval mode")
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--plant-n", type=int, default=10, help="answerable questions to repeat with a planted lie")
parser.add_argument("--limit", type=int, default=None, help="process at most N new questions in this run")
parser.add_argument("--run-name", default=None)
parser.add_argument("--sleep", type=float, default=2.0, help="pause between questions (seconds)")
parser.add_argument("--fresh", action="store_true", help="ignore and delete saved results for this run name")
parser.add_argument("--no-mlflow", action="store_true")
args = parser.parse_args()

config.RETRIEVAL_MODE = args.mode  # retrieve() reads this when it runs

# ---------- load and check the golden set ----------
try:
    items, problems = load_golden(args.golden)
except (OSError, json.JSONDecodeError) as e:
    sys.exit(f"Cannot read {args.golden}: {e}")
if problems:
    print("Fix these problems in the golden set first:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
missing_ref = [it["id"] for it in items if it["answerable"] and not str(it.get("answer", "")).strip()]
if missing_ref:
    sys.exit(f"These answerable questions have no reference 'answer': {missing_ref}")

golden_sha = hashlib.sha256(Path(args.golden).read_bytes()).hexdigest()[:10]
run_name = re.sub(r"[^A-Za-z0-9_.-]", "_", args.run_name or f"{args.mode}-verify")
results_dir = Path("eval/results")
results_dir.mkdir(parents=True, exist_ok=True)
cache_path = results_dir / f"agent_{run_name}.jsonl"
summary_path = results_dir / f"agent_{run_name}_summary.json"
md_path = results_dir / f"agent_{run_name}.md"

# ---------- decide what to run ----------
answerable = [it for it in items if it["answerable"]]
n_plant = min(args.plant_n, len(answerable))
plant_items = [answerable[int(i * len(answerable) / n_plant)] for i in range(n_plant)] if n_plant else []
tasks = [("normal", it) for it in items] + [("plant", it) for it in plant_items]
all_keys = [f"{kind}-{it['id']}" for kind, it in tasks]

if args.fresh and cache_path.exists():
    cache_path.unlink()
done = {}
if cache_path.exists():
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            done[rec["key"]] = rec
for rec in done.values():
    if (rec.get("golden"), rec.get("mode"), rec.get("k")) != (golden_sha, args.mode, args.k):
        sys.exit(
            "Saved results for this run name were made with a different golden set, mode or k.\n"
            "Use --fresh to start over, or pick another --run-name."
        )

llm_name = f"{config.LLM_PROVIDER}:{config.GROQ_MODEL if config.LLM_PROVIDER == 'groq' else config.GEMINI_MODEL}"
print(f"Run '{run_name}' | retrieval={args.mode} k={args.k} | LLM={llm_name}")
print(f"{len(tasks)} tasks in total ({len(items)} questions + {len(plant_items)} planted-lie checks), {len(done)} already saved")


def with_retry(fn, label):
    """Retry on short rate limits. Stop completely on a daily quota."""
    for attempt in range(1, 5):
        try:
            return fn()
        except Exception as e:
            text = str(e)
            if is_daily_limit(text):
                raise QuotaExhausted(text[:400]) from e
            if attempt == 4:
                raise
            wait = min((parse_wait(text) or 20 * attempt) + 2, 120)
            print(f"    ({label}: {type(e).__name__}, waiting {wait:.0f}s, retry {attempt}/3)", flush=True)
            time.sleep(wait)


def process(kind, it):
    plant = kind == "plant"
    t0 = time.time()
    state = with_retry(lambda: run(it["question"], args.k, None, plant), "agent")
    latency = round(time.time() - t0, 2)

    v_first, v_final = state["first_verification"], state["verification"]
    n1, f1 = count_flags(v_first)
    n2, f2 = count_flags(v_final)
    first, final = state["first_answer"], state["answer"]
    rec = {
        "key": f"{kind}-{it['id']}", "kind": kind, "id": it["id"], "question": it["question"],
        "answerable": it["answerable"], "mode": args.mode, "k": args.k, "golden": golden_sha,
        "first_answer": first, "final_answer": final, "rewrites": state["rewrites"],
        "first_sentences": n1, "first_flagged": f1, "final_sentences": n2, "final_flagged": f2,
        "first_trust": round(v_first["trust"], 4), "final_trust": round(v_final["trust"], 4),
        "refused": is_refusal(final), "source_pages": [c["page"] for c in state["chunks"]],
        "latency_s": latency,
    }
    if kind == "normal" and it["answerable"]:
        rec["retrieval_rank"] = first_relevant_rank(state["chunks"], it)
        rec["correct"], rec["correct_reason"] = with_retry(lambda: grade(it, final), "grading")
        if first == final:
            rec["first_correct"] = rec["correct"]
        else:
            rec["first_correct"], _ = with_retry(lambda: grade(it, first), "grading")
    if plant:
        rec["lie_detected"] = lie_detected(v_first)
        rec["lie_removed"] = lie_removed(final)
    return rec


def one_line(rec):
    base = f"trust {rec['first_trust']:.0%} -> {rec['final_trust']:.0%} | rewrites {rec['rewrites']} | {rec['latency_s']:.0f}s"
    if rec["kind"] == "plant":
        return f"{base} | lie detected={rec['lie_detected']} removed={rec['lie_removed']}"
    if rec["answerable"]:
        return f"{base} | {rec['correct']}"
    return f"{base} | refused={rec['refused']}"


# ---------- main loop ----------
todo = [(kind, it) for kind, it in tasks if f"{kind}-{it['id']}" not in done]
processed, failed, stopped = 0, 0, None
for kind, it in todo:
    if args.limit is not None and processed >= args.limit:
        break
    print(f"[{kind} #{it['id']}] {it['question'][:75]}", flush=True)
    try:
        rec = process(kind, it)
    except QuotaExhausted as e:
        stopped = str(e)
        break
    except Exception as e:
        failed += 1
        print(f"    FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        continue
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    done[rec["key"]] = rec
    processed += 1
    print("    " + one_line(rec), flush=True)
    time.sleep(args.sleep)

# ---------- results ----------
records = [done[k] for k in all_keys if k in done]
if not records:
    print("\nNo results yet.")
    if stopped:
        print("Stopped: the LLM daily quota looks used up. Details:\n  " + stopped)
    sys.exit(1)

complete = len(records) == len(all_keys)
summary = summarize(records)
meta = {
    "run_name": run_name, "mode": args.mode, "k": args.k, "llm": llm_name,
    "date": datetime.now().strftime("%Y-%m-%d"), "golden_sha": golden_sha,
    "tasks_done": len(records), "tasks_total": len(all_keys), "complete": complete,
}
table = render_markdown(summary, meta)
print("\n" + ("" if complete else "PARTIAL RESULTS (do not quote these yet)\n") + table)

print("Cases to inspect:")
shown = 0
for r in records:
    problems_here = []
    if r["kind"] == "normal" and r["answerable"] and r.get("correct") != "correct":
        problems_here.append(f"{r['correct']}: {r.get('correct_reason', '')[:110]}")
    if r["kind"] == "normal" and r["answerable"] and r.get("retrieval_rank") is None:
        problems_here.append(f"golden page not in retrieved passages (got pages {r['source_pages']})")
    if r["kind"] == "normal" and not r["answerable"] and not r["refused"]:
        problems_here.append("should have refused but answered")
    if r["kind"] == "normal" and r["final_flagged"] > 0:
        problems_here.append(f"{r['final_flagged']} sentence(s) still flagged in the final answer")
    if r["kind"] == "plant" and not (r["lie_detected"] and r["lie_removed"]):
        problems_here.append(f"planted lie detected={r['lie_detected']} removed={r['lie_removed']}")
    for text in problems_here:
        print(f"  [{r['kind']} #{r['id']}] {r['question'][:60]} -> {text}")
        shown += 1
if not shown:
    print("  none")

summary_path.write_text(json.dumps({"meta": meta, "summary": summary}, indent=2), encoding="utf-8")
md_path.write_text(table, encoding="utf-8")
print(f"\nSaved {summary_path} and {md_path}")

if stopped:
    print("\nSTOPPED: the LLM daily quota looks used up. Run the same command again later to continue.")
    print("  " + stopped[:300])
if failed:
    print(f"\n{failed} question(s) failed. Run the same command again to retry just those.")
if not complete:
    print(f"\n{len(records)} of {len(all_keys)} tasks done. Run the same command again to continue.")
    sys.exit(0)


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


if args.no_mlflow:
    print("\nCOMPLETE (MLflow logging skipped)")
    sys.exit(0)

try:
    import mlflow
except ImportError:
    sys.exit("\nCOMPLETE, but MLflow is not installed. Run: pip install -r requirements-eval.txt, then run this command again.")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("truthlens-eval")
with mlflow.start_run(run_name=run_name):
    mlflow.log_params({
        "llm": llm_name, "retrieval_mode": args.mode, "k": args.k, "candidates": config.CANDIDATES,
        "embed_model": config.EMBED_MODEL, "rerank_model": config.RERANK_MODEL,
        "max_rewrites": MAX_REWRITES, "golden_sha": golden_sha,
        "questions": len(items), "planted_lie_checks": len(plant_items),
    })
    mlflow.set_tag("git_commit", git_commit())
    mlflow.log_metrics({k: v for k, v in summary.items() if v is not None})
    for artifact in (summary_path, md_path, cache_path):
        mlflow.log_artifact(str(artifact))
print("\nCOMPLETE and logged to MLflow (experiment 'truthlens-eval').")
print("View it with:  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")