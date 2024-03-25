import os
import json
from src.lm_eval import run_lm_eval_benchmarks
from src.eq_bench import run_eq_bench_benchmarks, configure_eq_bench
from src.upload import upload_to_github_gist

def main():
	# Retrieve environment variables
	eq_bench_benchmarks_to_run = os.environ.get("EQ_BENCH_BENCHMARKS", "").split(",")
	eq_bench_benchmarks_to_run = [benchmark for benchmark in eq_bench_benchmarks_to_run if benchmark]
	
	lm_eval_tasks = os.environ["LM_EVAL_TASKS"].split(",")
	model_id = os.environ["MODEL_PATH"]
	trust_remote_code = os.environ["TRUST_REMOTE_CODE"] == "True"
	debug_mode = os.environ["DEBUG"] == "True"
	github_api_token = os.environ["GITHUB_API_TOKEN"]
	hf_api_token = os.environ["HF_API_TOKEN"]
	lm_eval_quantization = os.environ.get("LM_EVAL_QUANTIZATION", "none")
	lm_eval_batch_size = os.environ.get("LM_EVAL_BATCH_SIZE", "auto")
	lm_eval_log_samples = os.environ.get("LM_EVAL_LOG_SAMPLES", "False").lower() == "true"
	
	
	#if eq_bench_benchmarks_to_run:
		#configure_eq_bench(eq_bench_benchmarks_to_run)

	results = {
        "model_id": model_id,
        "lm_eval_results": None,
        "eq_bench_results": None
    }
	
	# Run benchmarks	

	if any(benchmark in eq_bench_benchmarks_to_run for benchmark in ["eq-bench", "creative-writing", "judgemark"]):
		results["eq_bench_results"] = run_eq_bench_benchmarks(eq_bench_benchmarks_to_run, model_id)
		print('EQ-Bench Completed.')
	
	
	if lm_eval_tasks:
		results["lm_eval_results"] = run_lm_eval_benchmarks(model_id, lm_eval_tasks, lm_eval_quantization, lm_eval_batch_size, trust_remote_code, hf_api_token, lm_eval_log_samples)
		print('LM-Eval Completed.')
	
	# Save the collated results to a JSON file
	sanitized_model_id = model_id.replace("/", "__")
	results_file = f"{sanitized_model_id}___EATT.json"
	with open(results_file, "w") as f:
		json.dump(results, f, indent=2)

	upload_to_github_gist(json.dumps(results), f"{sanitized_model_id}___EATT.json")
	print('Uploaded results to Gist.')
	print('--------------------')
	print('Benchmarks complete!')
	print('--------------------')

if __name__ == "__main__":
	main()