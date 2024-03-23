import subprocess
import configparser
import io

def run_command(pod, command: str) -> str:
	"""Run a command on the RunPod instance."""
	return pod.run_command(command).output

def parse_config(config_path: str) -> configparser.ConfigParser:
	"""Parse a configuration file."""
	config = configparser.ConfigParser()
	config.read(config_path)
	return config

def update_config(config: configparser.ConfigParser, options: dict):
	"""Update a configuration object with the given options."""
	for section, values in options.items():
		for key, value in values.items():
			config.set(section, key, value)

def write_config(config: configparser.ConfigParser, config_path: str):
	"""Write a configuration object to a file."""
	with open(config_path, "w") as config_file:
		config.write(config_file)