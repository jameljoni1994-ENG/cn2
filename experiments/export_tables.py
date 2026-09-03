"""Export metric tables (CSV) from results/benchmark.json."""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open("results/benchmark.json") as fh:
    data = json.load(fh)
rows = [r for r in data["rows"] if "error" not in r]

os.makedirs("results", exist_ok=True)

# 1) Hessian-eval table
probs = sorted({r["problem"] for r in rows})
methods = ["cn2", "newton", "newton_cg", "lbfgs", "nag", "gd"]

def table(metric):
    with open(f"results/table_{metric}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + probs)
        for m in methods:
            row = [m]
            for p in probs:
                rr = [r for r in rows if r["method"] == m and r["problem"] == p]
                row.append(f"{rr[0][metric]:.2e}" if rr else "ERR")
            w.writerow(row)

for met in ["final_f", "hess_evals", "grad_evals", "f_evals", "iters", "walltime", "cpu_s"]:
    table(met)
    print(f"wrote results/table_{met}.csv")

print("\nKey perspective — Hessians needed (CN² vs Newton) per problem:")
for p in probs:
    h_cn2 = next((r["hess_evals"] for r in rows if r["method"]=="cn2" and r["problem"]==p), None)
    h_nt  = next((r["hess_evals"] for r in rows if r["method"]=="newton" and r["problem"]==p), None)
    print(f"  {p:>20}: CN² hess={h_cn2}  Newton hess={h_nt}")
