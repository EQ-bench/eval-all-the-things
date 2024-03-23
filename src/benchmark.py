import os
import time
from typing import List
from .upload import upload_to_github_gist

def handle_results(lm_eval_results: dict, eq_bench_results: dict, elapsed_time: float):
	"""Handle the benchmark results."""
	github_api_token = os.environ["GITHUB_API_TOKEN"]
	
	summary = f"Benchmark Results:\n\n"
	
	if lm_eval_results:
		summary += "lm-eval Results:\n"
		summary += f"{lm_eval_results}\n\n"
	
	if eq_bench_results:
		summary += "eq-bench Results:\n"
		summary += f"{eq_bench_results}\n\n"
	
	summary += f"Elapsed Time: {elapsed_time:.2f} seconds"
	
	# Upload results to GitHub Gist
	gist_url = upload_to_github_gist(summary, "benchmark_results.txt")
	print(f"Benchmark results uploaded to GitHub Gist: {gist_url}")