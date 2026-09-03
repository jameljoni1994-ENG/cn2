These files are the OLD (superseded) benchmark runner and its output.

- runner_OLD.py            : superseded by experiments/run_all.py
- benchmark_results_OLD.json: produced by runner_OLD; uses DIFFERENT problems
                             and parameters (Rosen n=10, QuadIllCond, K0=20,
                             max_cycle=5000) so its numbers do NOT match the
                             paper. Kept only for reference.

The authoritative results are:
  experiments/run_all.py        -> results/benchmark.json
                                  results/table_*.csv
                                  figures/*.png
  paper/cn2_paper_en.tex (pdf)  : English paper
  paper/cn2_paper_ar.tex (pdf)  : Arabic paper
