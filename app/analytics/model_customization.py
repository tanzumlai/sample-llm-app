import datahub.emitter.mce_builder as builder
import datahub.metadata.schema_classes as models
from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.emitter.rest_emitter import DatahubRestEmitter
from huggingface_hub import ModelCard
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, TFAutoModelForQuestionAnswering
from dotenv import load_dotenv
import os
import mlflow
from mlflow import MlflowClient
from transformers import pipeline
import mlflow.pyfunc
import logging
import traceback
import requests
from huggingface_hub import create_repo
from app.analytics import config
import torch

load_dotenv()


def send_metadata(model_name: str,
                  platform: str,
                  env: str,
                  gms_server: str,
                  model_description: str = None):
    logging.info("Updating metadata catalog...")
    with mlflow.start_run(run_name='send_metadata', nested=True):
        emitter = DatahubRestEmitter(gms_server=gms_server, extra_headers={})
        model_urn = builder.make_ml_model_urn(
            model_name=model_name, platform=platform, env=env
        )
        model_card = ingest_metadata_from_huggingface_model(model_name)

        metadata_change_proposal = MetadataChangeProposalWrapper(
            entityType="mlModel",
            changeType=models.ChangeTypeClass.UPSERT,
            entityUrn=model_urn,
            aspectName="mlModelProperties",
            aspect=models.MLModelPropertiesClass(
                description=model_card.text,
                customProperties={**{k: ','.join(str(v)) for (k, v) in model_card.data.to_dict().items()},
                                  **{'Last Updated': ''}})
    )

    emitter.emit(metadata_change_proposal)


def ingest_metadata_from_huggingface_model(model_name: str):
    card = ModelCard.load(model_name)
    return card or {}


def publish_model(repo_name: str, pretrained_model_name: str):
    logging.info("Publishing model to Hub...")
    with mlflow.start_run(run_name='publish_model', nested=True):
        model_name = repo_name

        logging.info("=====================\nCreating model repo if it does not exist...\n=====================\n")
        create_repo(model_name, token=os.getenv('DATA_E2E_HUGGINGFACE_TOKEN'), exist_ok=True)

        logging.info(f"=====================\nSaving model {model_name}...\n=====================\n")
        model = _select_base_llm_class(pretrained_model_name)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)
        model.save_pretrained(pretrained_model_name)
        tokenizer.save_pretrained(pretrained_model_name)
        model.push_to_hub(model_name, max_shard_size='2GB', token=os.getenv('DATA_E2E_HUGGINGFACE_TOKEN'))
        tokenizer.push_to_hub(model_name, token=os.getenv('DATA_E2E_HUGGINGFACE_TOKEN'))


def promote_model_to_staging(model_name, pipeline_name, persist_model_copy: str):
    logging.info("Promoting model to staging...")
    with mlflow.start_run(run_name='promote_model_to_staging', nested=True) as run:
        client = MlflowClient()

        qa_pipe = pipeline(pipeline_name, model_name)
        # If persist_model_copy is "yes", save a version of the model which stores a full copy in the model registry
        if persist_model_copy == "yes":
            logging.info("Saving full copy of the model...")
            mlflow.transformers.log_model(
                transformers_model=qa_pipe,
                artifact_path=pipeline_name,
            )

        # Save a version of the model which stores a link to the source model repo
        logging.info("Saving link to source model repo...")
        mlflow.transformers.log_model(
            transformers_model=qa_pipe,
            artifact_path=pipeline_name,
            save_pretrained=False,
        )

        registered_model_name = model_name.replace('/', '-')
        model_uri = f"runs:/{run.info.run_id}/{pipeline_name}"
        mlflow.register_model(model_uri, registered_model_name)
        mv = client.create_model_version(registered_model_name, model_uri, run.info.run_id)

        # Promote to staging
        mv_copy = client.copy_model_version(
            src_model_uri=f"models:/{registered_model_name}/{mv.version}",
            dst_name=f"{registered_model_name}-staging",
        )

        # Set up alias
        client.set_registered_model_alias(
            name=f"{registered_model_name}-staging",
            alias="champion",
            version=mv_copy.version,
        )


def select_base_llm():
    # If a default LLM exists in the model registry, return it
    try:
        logging.info("Retrieving default production model from ML registry if exists...")
        model_name = f"{config.model_name}-{config.model_stage.lower()}"
        model_api_uri = f'{os.getenv("MLFLOW_TRACKING_URI")}/api/2.0/mlflow/registered-models/alias?name={model_name}&alias={config.model_alias}'
        models = requests.get(model_api_uri).json()
        if 'model_version' in models:
            if 'source' in models['model_version']:
                return models['model_version'].get('source')
            elif 'source_model_name' in models['model_version']:
                return models['model_version'].get('source_model_name')
    except Exception as e:
        logging.error(f"Registered model name={model_name}, alias={config.model_alias} not found.")
        logging.info(str(e))
        logging.info(''.join(traceback.TracebackException.from_exception(e).format()))

    # Else, return a predefined default model
    default_model = config.fallback_model_name
            
    return default_model


def _select_base_llm_class(pretrained_model_name):
    for m in [AutoModelForCausalLM, TFAutoModelForQuestionAnswering]:
        try:
            model = m.from_pretrained(pretrained_model_name,
                                      torch_dtype=torch.bfloat16,
                                      device_map='cpu',
                                      low_cpu_mem_usage=True,
                                      trust_remote_code=True)
            return model
        except Exception as e:
            logging.info(str(e))
            logging.info(''.join(traceback.TracebackException.from_exception(e).format()))
            continue
    logging.error("Could not load task-specific base class, will use AutoModel...")
    return AutoModel



