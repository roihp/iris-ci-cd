# deploy.py
import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential

import uuid

endpoint_name = "iris-endpoint-" + str(uuid.uuid4())[:8]
deployment_name = "blue"

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_WORKSPACE_NAME"]
)

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    auth_mode="key",
)

ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Create environment
env = Environment(
    name="iris-env",
    conda_file="conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)

ml_client.environments.create_or_update(env)

model_name = "iris-model"
model = ml_client.models.get(name=model_name, version=None)  # None = latest

# Create deployment
deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=endpoint_name,
    model=f"{model.name}:{model.version}",
    environment=env,
    code_configuration=CodeConfiguration(
        code="./",
        scoring_script="score.py"
    ),
    instance_type="Standard_DS3_v2",
    instance_count=1,
)

ml_client.online_deployments.begin_create_or_update(deployment).result()

# Set traffic
ml_client.online_endpoints.begin_update(
    ManagedOnlineEndpoint(
        name=endpoint_name,
        traffic={"blue": 100}
    )
).result()

print("Deployment successful!")
