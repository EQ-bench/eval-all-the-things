import os
import subprocess
from typing import List
import json
import glob

def install_lm_eval_dependencies():
	"""Install lm-eval dependencies."""
	commands = [
		"apt update -y",
		"apt install nano screen htop -y",
		"git clone https://github.com/sqrkl/lm-evaluation-harness -b mmlu-pro-irt",
		#"git clone https://github.com/sqrkl/lm-evaluation-harness -b many-shot-testing",
  		#"git clone https://github.com/sqrkl/lm-evaluation-harness",
		"cd lm-evaluation-harness",
		"pip install -e .",
		"pip install gekko sentencepiece hf_transfer einops optimum accelerate bitsandbytes tiktoken flash_attn transformers_stream_generator tokenizers",
		"pip install git+https://github.com/huggingface/transformers.git",
		"export HF_HUB_ENABLE_HF_TRANSFER=1",
		"export NUMEXPR_MAX_THREADS=64"
	]	
	subprocess.run(' && '.join(commands), shell=True, check=True)

def get_results_files(full_output_dir):
	# Search for the JSON file with the pattern results_*.json
	json_pattern = os.path.join(full_output_dir, "results_*.json")
	json_files = glob.glob(json_pattern)
	return json_files

def run_lm_eval_benchmarks(model_id: str, tasks: List[str], quantization: str, batch_size: str, trust_remote_code: bool, hf_api_token: str, log_samples: bool):
	"""Run lm-eval benchmarks."""
	install_lm_eval_dependencies()

	for i,t in enumerate(tasks):
		tasks[i] = t.strip()

	quant_args = ",dtype=bfloat16"
	if quantization == "4bit":
		quant_args = ",load_in_4bit=True,bnb_4bit_compute_dtype=float16"
	elif quantization == "8bit":
		quant_args = ",load_in_8bit=True"

	#output_dir = f"output/{model_id.replace('/', '__')}"
	output_dir = 'output'
	full_output_dir = os.path.join("output", model_id.replace('/', '__'))
	os.makedirs(output_dir, exist_ok=True)

	openelm_params = ""
	if model_id.startswith('apple/OpenELM'):
		openelm_params = ',add_bos_token=True,tokenizer=NousResearch/Llama-2-7b-hf'

	hf_login_cmd = ''
	if os.environ["HF_API_TOKEN"]:
		hf_login_cmd = 'huggingface-cli login --token ' + os.environ["HF_API_TOKEN"] + ' && '

	log_samples_arg = "--log_samples" if log_samples else ""
	command_template = f"{hf_login_cmd}export HF_HUB_ENABLE_HF_TRANSFER=1 && export NUMEXPR_MAX_THREADS=64 && lm_eval --model hf --model_args pretrained={model_id},device_map=auto,max_length=4096,trust_remote_code={trust_remote_code}{openelm_params}{quant_args} --tasks {','.join(tasks)} --device auto --batch_size {batch_size} --output_path {output_dir} --use_cache sqlite_cache_{model_id.replace('/', '__')} --verbosity DEBUG {log_samples_arg}"

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

	# Retry logic
	while not get_results_files(full_output_dir):
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

	

	# Load the JSON file if it exists

	results["lm_eval_results"] = []	
	for json_file in get_results_files(full_output_dir):		
		with open(json_file, "r") as f:
			results["lm_eval_results"].append(json.load(f))

	# Parse the sample JSONL files
	if log_samples:
		for file in os.listdir(full_output_dir):
			if file.endswith(".jsonl"):
				with open(os.path.join(full_output_dir, file), "r") as f:
					results["lm_eval_samples"].append(json.load(f))

	return results