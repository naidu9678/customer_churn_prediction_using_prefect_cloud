
import requests

# Replace these variables with your actual Prefect Cloud credentials
PREFECT_API_KEY = "pnu_WN525BrLmbiIO3srOrti5qLCeFrsx113bY0z"  # Your Prefect Cloud API key
ACCOUNT_ID = "e5a12ba2-a3dc-4d0f-a42c-4b0459f708e9"  # Your Prefect Cloud Account ID
WORKSPACE_ID = "b073822f-9781-46af-912e-635ec2b0c441"  # Your Prefect Cloud Workspace ID

# Correct API URL to list flow runs
PREFECT_API_URL = f"https://api.prefect.cloud/api/accounts/{ACCOUNT_ID}/workspaces/{WORKSPACE_ID}/artifacts/filter"

# Data to filter artifacts
data = {
    "sort": "CREATED_DESC",
    "limit": 5,
    "artifacts": {
        "key": {
            "exists_": True
        }
    }
}

# Set up headers with Authorization
headers = {"Authorization": f"Bearer {PREFECT_API_KEY}"}

# Make the request
response = requests.post(PREFECT_API_URL, headers=headers, json=data)
print(response)

# Check the response status
if response.status_code != 200:
    print(f"Error: Received status code {response.status_code}")
    print(f"Response content: {response.text}")
else:
    artifacts = response.json()
    print(artifacts)
    for artifact in artifacts:
        print(artifact)
