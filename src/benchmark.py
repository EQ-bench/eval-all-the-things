import os
import time
from typing import List
from runpod import RunPod
from .lm_eval import run_lm_eval_benchmarks
from .eq_bench import run_eq_bench_benchmarks
from .upload import upload_to_github_gist

def launch_runpod_instance(gpu_type: str, num_gpus: int, disk_size: int, cloud_type: str, repo_url: str, 
								trust_remote_code: bool, debug_mode: bool) -> RunPod:
	"""Launch a RunPod instance with the specified configuration."""
	runpod = RunPod(api_key=os.environ["RUNPOD_API_KEY"])
	pod = runpod.create_pod(
		gpu_type=gpu_type,
		num_gpus=num_gpus,
		disk_size=disk_size,
		cloud_type=cloud_type,
		repo_url=repo_url,
		env={
			"TRUST_REMOTE_CODE": str(trust_remote_code),
			"DEBUG_MODE": str(debug_mode)
		}
	)
	return pod

def install_dependencies(pod: RunPod):
	"""Install the necessary dependencies on the RunPod instance."""
	pod.run_command("pip install -r requirements.txt")

def run_benchmarks(pod: RunPod, model_id: str, lm_eval_tasks: List[str], eq_bench_options: dict):
	"""Run the selected benchmarks on the RunPod instance."""
	start_time = time.time()

	"""Run the selected benchmarks on the RunPod instance."""
	benchmarks = os.environ["BENCHMARKS"].split(",")
	trust_remote_code = os.environ["TRUST_REMOTE_CODE"] == "True"
	debug_mode = os.environ["DEBUG"] == "True"
	hf_api_token = os.environ["HF_API_TOKEN"]
	
	if lm_eval_tasks:
		lm_eval_results = run_lm_eval_benchmarks(pod, model_id, lm_eval_tasks)
	else:
		lm_eval_results = None
	
	if any(option in eq_bench_options for option in ["eq_bench", "creative_writing", "judgemark"]):
		eq_bench_results = run_eq_bench_benchmarks(pod, eq_bench_options)
	else:
		eq_bench_results = None
	
	end_time = time.time()
	elapsed_time = end_time - start_time
	
	return lm_eval_results, eq_bench_results, elapsed_time

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