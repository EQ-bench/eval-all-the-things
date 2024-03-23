from typing import List
from .utils import run_command

def install_lm_eval_dependencies(pod):
	"""Install lm-eval dependencies on the RunPod instance."""
	commands = [
		"apt update -y",
		"apt install nano screen htop -y",
		"git clone https://github.com/sqrkl/lm-evaluation-harness",
		"cd lm-evaluation-harness",
		"pip install -e .",
		"pip install gekko sentencepiece hf_transfer einops optimum accelerate bitsandbytes tiktoken flash_attn transformers_stream_generator",
		"export HF_HUB_ENABLE_HF_TRANSFER=1",
		"export NUMEXPR_MAX_THREADS=64",
		"huggingface-cli login --token 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"
	]
	for command in commands:
		run_command(pod, command)

def run_lm_eval_benchmarks(pod, model_id: str, tasks: List[str]):
	"""Run lm-eval benchmarks on the RunPod instance."""
	install_lm_eval_dependencies(pod)
	
	command = f"lm_eval --model hf --model_args pretrained={model_id},trust_remote_code=True --tasks {','.join(tasks)} --device cuda:0 --batch_size auto --output_path ./lm_eval_results.json"
	output = run_command(pod, command)
	
	# Parse lm-eval results
	lm_eval_results = pod.read_file("./lm_eval_results.json")
	
	return lm_eval_results