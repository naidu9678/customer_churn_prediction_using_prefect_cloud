
import requests

# Replace these variables with your actual Prefect Cloud credentials
PREFECT_API_KEY = "xxxx"  # Your Prefect Cloud API key
ACCOUNT_ID = "xxxxx"  # Your Prefect Cloud Account ID
WORKSPACE_ID = "xxxx"  # Your Prefect Cloud Workspace ID
DEPLOYMENT_ID = "xxxx"  # Your Deployment ID

# Correct API URL to get deployment details
PREFECT_API_URL = f"https://api.prefect.cloud/api/accounts/{ACCOUNT_ID}/workspaces/{WORKSPACE_ID}/deployments/{DEPLOYMENT_ID}"

# Set up headers with Authorization
headers = {"Authorization": f"Bearer {PREFECT_API_KEY}"}

# Make the request using GET
response = requests.get(PREFECT_API_URL, headers=headers)

# Check the response status
if response.status_code == 200:
    deployment_info = response.json()
    print(deployment_info)
else:
    print(f"Error: Received status code {response.status_code}")
    print(f"Response content: {response.text}")
