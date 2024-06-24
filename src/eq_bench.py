import configparser
import io
import os
import subprocess
import json

def install_eq_bench_dependencies():
	"""Install eq-bench dependencies."""
	if os.environ["INFERENCE_ENGINE"] == 'ooba':
		commands = [
			'REPO_DIR="EQ-Bench"; if [ -d "$REPO_DIR" ]; then cd "$REPO_DIR" && git pull; else git clone -b main_v2_4 https://github.com/EQ-bench/EQ-Bench.git; cd "$REPO_DIR"; fi; ./ooba_quick_install.sh'
		]
	else:
		commands = [
			'REPO_DIR="EQ-Bench"; if [ -d "$REPO_DIR" ]; then cd "$REPO_DIR" && git pull; else git clone -b main_v2_4 https://github.com/EQ-bench/EQ-Bench.git; cd "$REPO_DIR"; fi; ./install_reqs.sh'
		]
	subprocess.run(' && '.join(commands), shell=True, check=True)

	# Save Firebase credentials if provided
	firebase_creds_env = os.environ.get("FIREBASE_CREDS")
	if firebase_creds_env:
		# Reverse the previous replacements to get the original JSON string
		firebase_creds_json_str = firebase_creds_env.replace('<NL>', '\\n').replace('\\"', '"')

		with open("EQ-Bench/firebase_creds.json", "w") as f:
			f.write(firebase_creds_json_str)

def configure_eq_bench():
	config = configparser.ConfigParser(allow_no_value=True)
	config.optionxform = str  # Preserve case sensitivity of keys

	# Read the existing config file
	with open("EQ-Bench/config.cfg", "r") as config_file:
		config.read_file(config_file)

	if os.environ["INFERENCE_ENGINE"] == 'ooba':
		config.set('Oobabooga config', 'ooba_launch_script', '~/text-generation-webui/start_linux.sh')
	else:
		config.set('Oobabooga config', 'automatically_launch_ooba', 'false')

	if os.environ["HF_API_TOKEN"]:
		config.set('Huggingface', 'access_token', os.environ["HF_API_TOKEN"])

	# Update the configuration based on the provided options
	#for section, options in eq_bench_options.items():
	#	for key, value in options.items():
	#		config.set(section, key, value)

	# Retrieve benchmark run details from environment variables
	run_id = '1'  # os.environ["RUN_ID"]
	instruction_template = os.environ.get("INSTRUCTION_TEMPLATE", "")
	model_path = os.environ.get("MODEL_PATH", "")
	lora_path = os.environ.get("LORA_PATH", "")
	quantization = os.environ.get("QUANTIZATION", "")
	n_iterations = os.environ["N_ITERATIONS"]
	inference_engine = os.environ["INFERENCE_ENGINE"]
	ooba_params = os.environ.get("OOBA_PARAMS", "")
	downloader_filters = os.environ.get("DOWNLOADER_FILTERS", "")

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
	configure_eq_bench()

	# Change to the EQ-Bench directory
	os.chdir("EQ-Bench")

	results = {
		"model_id": model_id,
		"eq_bench_results": {},
		"eq_bench_output": ""
	}

	# Run selected eq-bench benchmarks
	for benchmark in benchmarks:
		command = f"python eq-bench.py -v -f --benchmarks {benchmark}"
		# Start the subprocess and get its output stream
		process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)
		
		print(f"\n\n=== {benchmark} ===")  # Display benchmark header
		results["eq_bench_output"] += f"\n\n=== {benchmark} ===\n"  # Save benchmark header
		
		# Read the output line by line as it becomes available
		for line in iter(process.stdout.readline, ''):
			print(line, end='')  # Display output line without adding extra newline
			results["eq_bench_output"] += line  # Save output line

		process.stdout.close()
		return_code = process.wait()
		#if return_code:
		#	raise subprocess.CalledProcessError(return_code, command)

	# Parse eq-bench results
	try:
		with open("raw_results.json", "r") as f:
			results["eq_bench_results"]["raw_results.json"] = json.load(f)
	except Exception as e:
		print(e)

	try:
		with open("benchmark_results.csv", "r") as f:
			results["eq_bench_results"]["benchmark_results.csv"] = f.read()
	except Exception as e:
		print(e)

	# Change back to the parent directory
	os.chdir("..")

	return results