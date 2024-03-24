from .utils import run_command
import configparser
import io
import os
import subprocess
import json

def install_eq_bench_dependencies(pod):
	"""Install eq-bench dependencies on the RunPod instance."""
	commands = [
		"git clone https://github.com/EQ-bench/EQ-Bench.git",
		"cd EQ-Bench",
		"./ooba_quick_install.sh"
	]
	for command in commands:
		run_command(pod, command)

def configure_eq_bench(eq_bench_options):
	config = configparser.ConfigParser(allow_no_value=True)
	config.optionxform = str  # Preserve case sensitivity of keys

	# Read the existing config file
	with open("EQ-Bench/config.cfg", "r") as config_file:
		config.read_file(config_file)

	# Update the configuration based on the provided options
	for section, options in eq_bench_options.items():
		for key, value in options.items():
			config.set(section, key, value)

	# Retrieve benchmark run details from environment variables
	run_id = os.environ["RUN_ID"]
	instruction_template = os.environ["INSTRUCTION_TEMPLATE"]
	model_path = os.environ["MODEL_PATH"]
	lora_path = os.environ["LORA_PATH"]
	quantization = os.environ["QUANTIZATION"]
	n_iterations = os.environ["N_ITERATIONS"]
	inference_engine = os.environ["INFERENCE_ENGINE"]
	ooba_params = os.environ["OOBA_PARAMS"]
	downloader_filters = os.environ["DOWNLOADER_FILTERS"]

	# Update the benchmark run configuration
	benchmark_run = f"{run_id}, {instruction_template}, {model_path}, {lora_path}, {quantization}, {n_iterations}, {inference_engine}, {ooba_params}, {downloader_filters}"	

	# Write the updated configuration back to the file
	with open("EQ-Bench/config.cfg", "w") as config_file:
		config.write(config_file)
		config_file.write('\n'+benchmark_run)
	

def run_eq_bench_benchmarks(benchmarks, model_id):
	# Install eq-bench dependencies
	install_eq_bench_dependencies()

	# Configure eq-bench
	configure_eq_bench(benchmarks)

	results = {
		"model_id": model_id,
		"eq_bench_results": {
			'raw_results.json': None,
			'benchmark_results.csv': None
		},
		"eq_bench_output": ""
	}

	# Run selected eq-bench benchmarks
	for benchmark in benchmarks:
		command = f"python EQ-Bench/eq-bench.py --benchmarks {benchmark}"
		output = subprocess.check_output(command, shell=True).decode("utf-8")
		results["eq_bench_output"] += f"\n\n=== {benchmark} ===\n{output}"

	# Parse eq-bench results
	try:
		with open("EQ-Bench/raw_results.json", "r") as f:
			results["eq_bench_results"]['raw_results.json'] = json.load(f)
	except Exception as e:
		print(e)
	try:
		with open("EQ-Bench/benchmark_results.csv", "r") as f:
			results["eq_bench_results"]['benchmark_results.csv'] = json.load(f)
	except Exception as e:
		print(e)

	return results