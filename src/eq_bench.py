from .utils import run_command
import configparser
import io

def install_eq_bench_dependencies(pod):
	"""Install eq-bench dependencies on the RunPod instance."""
	commands = [
		"git clone https://github.com/EQ-bench/EQ-Bench.git",
		"cd EQ-Bench",
		"./ooba_quick_install.sh"
	]
	for command in commands:
		run_command(pod, command)

def configure_eq_bench(pod, options: dict):
	"""Configure eq-bench options on the RunPod instance."""
	config_content = pod.read_file("EQ-Bench/config.cfg")
	config = configparser.ConfigParser()
	config.read_string(config_content)
	
	for section, values in options.items():
		for key, value in values.items():
			config.set(section, key, value)
	
	updated_config_content = io.StringIO()
	config.write(updated_config_content)
	pod.write_file("EQ-Bench/config.cfg", updated_config_content.getvalue())

def run_eq_bench_benchmarks(pod, options: dict):
	"""Run eq-bench benchmarks on the RunPod instance."""
	install_eq_bench_dependencies(pod)
	configure_eq_bench(pod, options)
	
	benchmarks = []
	if options.get("eq_bench", False):
		benchmarks.append("eq-bench")
	if options.get("creative_writing", False):
		benchmarks.append("creative-writing")
	if options.get("judgemark", False):
		benchmarks.append("judgemark")
	
	command = f"python EQ-Bench/eq-bench.py --benchmarks {' '.join(benchmarks)}"
	output = run_command(pod, command)
	
	# Parse eq-bench results
	eq_bench_results = pod.read_file("EQ-Bench/results.json")
	
	return eq_bench_results