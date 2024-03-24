import os
import requests

def upload_to_github_gist(content: str, filename: str) -> str:
	"""Upload content to a GitHub Gist."""
	url = "https://api.github.com/gists"
	headers = {
		"Authorization": f"token {os.environ['GITHUB_TOKEN']}",
		"Accept": "application/vnd.github.v3+json"
	}
	data = {
		"description": "Benchmark Results",
		"public": True,
		"files": {
			filename: {
					"content": content
			}
		}
	}
	response = requests.post(url, headers=headers, json=data)
	response.raise_for_status()
	gist_url = response.json()["html_url"]
	return gist_url

def upload_to_gist(file_path, github_token):
	with open(file_path, "r") as f:
		content = f.read()

	data = {
		"description": "EQ-Bench Results",
		"public": True,
		"files": {
			file_path: {
					"content": content
			}
		}
	}

	headers = {
		"Authorization": f"token {github_token}",
		"Accept": "application/vnd.github.v3+json"
	}

	response = requests.post("https://api.github.com/gists", json=data, headers=headers)
	response.raise_for_status()

	gist_url = response.json()["html_url"]
	print(f"Results uploaded to Gist: {gist_url}")