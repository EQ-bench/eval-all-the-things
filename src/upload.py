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