import os
import json
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml.entities import Model, Environment, CodeConfiguration

# credential = DefaultAzureCredential()

# Load GitHub secret directly
creds = json.loads(base64.b64decode(os.environ["AZURE_CREDENTIALS_BASE64"]))

credential = ClientSecretCredential(
    tenant_id=creds["tenantId"],
    client_id=creds["clientId"],
    client_secret=creds["clientSecret"]
)

ml_client = MLClient(
    credential,
    "691cbb21-a34e-4990-99a9-4bb74b409c18",
    "rhp-ml-rg2",
    "rhp-ml-workspace2",
)

# Register model
model = ml_client.models.create_or_update(
    Model(
        path="outputs/model.pkl",
        name="iris-model"
    )
)

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="rhp-iris-endpoint",
    auth_mode="key"
)

ml_client.begin_create_or_update(endpoint).wait()

# Deployment
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="rhp-iris-endpoint",
    model=model,
    environment=Environment(
        conda_file="conda.yaml",
        image="mcr.microsoft.com/azureml/sklearn-1.0-ubuntu20.04-py38-cpu"
    ),
    code_configuration=CodeConfiguration(
        code=".",
        scoring_script="score.py"
    ),
    instance_type="Standard_DS3_v2",
    instance_count=1
)

ml_client.begin_create_or_update(deployment).wait()

# IMPORTANT — route traffic
endpoint = ml_client.online_endpoints.get("rhp-iris-endpoint")
endpoint.traffic = {"blue": 100}
ml_client.begin_create_or_update(endpoint).wait()
