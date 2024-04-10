import os
import subprocess
from typing import List
import json

def install_lm_eval_dependencies():
	"""Install lm-eval dependencies."""
	commands = [
		"apt update -y",
		"apt install nano screen htop -y",
		"git clone https://github.com/sqrkl/lm-evaluation-harness",
		"cd lm-evaluation-harness",
		"pip install -e .",
		"pip install gekko sentencepiece hf_transfer einops optimum accelerate bitsandbytes tiktoken flash_attn transformers_stream_generator git+https://github.com/huggingface/transformers.git",
		"export HF_HUB_ENABLE_HF_TRANSFER=1",
		"export NUMEXPR_MAX_THREADS=64"
	]	
	subprocess.run(' && '.join(commands), shell=True, check=True)

def run_lm_eval_benchmarks(model_id: str, tasks: List[str], quantization: str, batch_size: str, trust_remote_code: bool, hf_api_token: str, log_samples: bool):
	"""Run lm-eval benchmarks."""
	install_lm_eval_dependencies()

	for i,t in enumerate(tasks):
		tasks[i] = t.strip()

	quant_args = ""
	if quantization == "4bit":
		quant_args = ",load_in_4bit=True,bnb_4bit_compute_dtype=float16"
	elif quantization == "8bit":
		quant_args = ",load_in_8bit=True"

	output_dir = f"output/{model_id.replace('/', '__')}"
	os.makedirs(output_dir, exist_ok=True)

	hf_login_cmd = ''
	if os.environ["HF_API_TOKEN"]:
		hf_login_cmd = 'huggingface-cli login --token ' + os.environ["HF_API_TOKEN"] + ' && '

	log_samples_arg = "--log_samples" if log_samples else ""
	command_template = f"{hf_login_cmd}export HF_HUB_ENABLE_HF_TRANSFER=1 && export NUMEXPR_MAX_THREADS=64 && lm_eval --model hf --model_args pretrained={model_id},trust_remote_code={trust_remote_code}{quant_args} --tasks {','.join(tasks)} --device auto --batch_size {batch_size} --output_path {output_dir}/lm_eval_results.json --use_cache sqlite_cache_{model_id.replace('/', '__')} --verbosity DEBUG {log_samples_arg}"

	def generate_command(current_batch_size):
		return command_template.replace("{batch_size}", current_batch_size)

	def run_benchmark(current_batch_size):
		final_command = generate_command(current_batch_size)
		output = subprocess.check_output(final_command, shell=True, env={"HF_API_TOKEN": hf_api_token}).decode("utf-8")
		return output

	#current_batch_size = batch_size
	#output = run_benchmark(current_batch_size)

	# Retry logic
	current_batch_size = batch_size if batch_size else 'auto:9'
	output = run_benchmark(current_batch_size)

	results_files = [f for f in os.listdir(output_dir) if f.endswith("results.json")]

	# Retry logic
	while not results_files:
		if current_batch_size == '1':
			break  # Stop if batch size is already at minimum
		current_batch_size = str(int(current_batch_size) // 2 if current_batch_size.isdigit() else 8)
		output = run_benchmark(current_batch_size)

	# Collate the results
	results = {
		"model_id": model_id,
		"lm_eval_results": {},
		"lm_eval_samples": [],
		"lm_eval_output": output
	}

	# Parse the overall results JSON file
	results_file = os.path.join(output_dir, "lm_eval_results.json")
	if os.path.exists(results_file):
		with open(results_file, "r") as f:
			results["lm_eval_results"] = json.load(f)

	# Parse the sample JSONL files
	if log_samples:
		for file in os.listdir(output_dir):
			if file.endswith(".jsonl"):
				with open(os.path.join(output_dir, file), "r") as f:
					results["lm_eval_samples"].append(json.load(f))

	return results