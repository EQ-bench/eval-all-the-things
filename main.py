# main.py

import os
from src.benchmark import handle_results
from src.lm_eval import run_lm_eval_benchmarks
from src.eq_bench import run_eq_bench_benchmarks, configure_eq_bench

def main():
	# Retrieve environment variables
	benchmarks = os.environ["BENCHMARKS"].split(",")
	lm_eval_tasks = os.environ["LM_EVAL_TASKS"].split(",")
	model_id = os.environ["MODEL_ID"]
	trust_remote_code = os.environ["TRUST_REMOTE_CODE"] == "True"
	debug_mode = os.environ["DEBUG"] == "True"
	github_api_token = os.environ["GITHUB_API_TOKEN"]
	hf_api_token = os.environ["HF_API_TOKEN"]
	
	# Configure eq-bench options
	eq_bench_options = {
		"Benchmarks to run": {
			"eq_bench": "eq-bench" in benchmarks,
			"creative_writing": "creative-writing" in benchmarks,
			"judgemark": "judgemark" in benchmarks
		}
	}
	configure_eq_bench(eq_bench_options)
	
	# Run benchmarks
	lm_eval_results = None
	eq_bench_results = None
	
	if lm_eval_tasks:
		lm_eval_results = run_lm_eval_benchmarks(model_id, lm_eval_tasks, trust_remote_code)
	
	if any(benchmark in benchmarks for benchmark in ["eq-bench", "creative-writing", "judgemark"]):
		eq_bench_results = run_eq_bench_benchmarks(model_id, trust_remote_code)
	
	# Handle results
	handle_results(lm_eval_results, eq_bench_results, github_api_token)

if __name__ == "__main__":
	main()