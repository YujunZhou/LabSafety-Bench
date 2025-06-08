# Configuration management
import os

# Set all environment variables at once
os.environ.update({
    "OPENAI_API_KEY": "",
    "GOOGLE_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
})

# Add DeepInfra configuration
os.environ["DEEPINFRA_API_KEY"] = ""

# Add LLaMA-Factory fine-tuned model paths
LLAMAFACTORY_PATH = "../../../LLaMA-Factory"
FINETUNED_MODELS_PATH = os.path.join(LLAMAFACTORY_PATH, "saves/llama3-8b/lora/labsafety")

model_path_dicts = {
    "vicuna": "lmsys/vicuna-7b-v1.5",
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
    'llama3-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    "vicuna-13b": "lmsys/vicuna-13b-v1.5",
    'galactica': 'facebook/galactica-6.7b',
    'darwin': 'darwin-7b',
    'llama3.1': 'meta-llama/Llama-3.1-8B-Instruct',
    'claude3.5-sonnet': 'claude-3-5-sonnet-20240620',
    'claude3-haiku': 'claude-3-haiku-20240307',
    'o1-mini': 'o1-mini',
    'deepseek-r1': "deepseek-ai/DeepSeek-R1",
    'llama3.3-70b': "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    'mistral-8x7b': "mistralai/Mixtral-8x7B-Instruct-v0.1",
    
    # Add Fine-tuned models
    'labsafety-scenario': os.path.join(FINETUNED_MODELS_PATH, "scenario_hazards"),
    'labsafety-decision': os.path.join(FINETUNED_MODELS_PATH, "decision_consequence"),
    'labsafety-decision2': os.path.join(FINETUNED_MODELS_PATH, "decision_consequence2"),
    'labsafety-decision1': os.path.join(FINETUNED_MODELS_PATH, "decision_consequence1"),
    'labsafety-text-qa': os.path.join(FINETUNED_MODELS_PATH, "text_qa"),
    'labsafety-scenario-decision': os.path.join(FINETUNED_MODELS_PATH, "scenario_decision"),
    'labsafety-scenario-qa': os.path.join(FINETUNED_MODELS_PATH, "scenario_qa"),
    'labsafety-decision-qa': os.path.join(FINETUNED_MODELS_PATH, "decision_qa"),
    'labsafety-all': os.path.join(FINETUNED_MODELS_PATH, "all_combined"),
    'labsafety-text-qa-dpo': os.path.join(FINETUNED_MODELS_PATH, "text_qa_dpo"),
}

# Model type classification: proprietary async models and local batch processing models
async_models = ['gpt-4o-mini', 'gpt-4o-2024-08-06', 'gpt-4o', 'gemini-1.5-flash', 'gemini-1.5-pro', 'claude3-haiku', 'claude3.5-sonnet', 'o3-mini', 'gemini-2.0-flash', 'deepseek-r1', 'llama3.3-70b', 'mistral-8x7b']
local_models = ['mistral', 'llama3-instruct', 'vicuna', 'vicuna-13b', 
                'labsafety-scenario', 'labsafety-decision', 'labsafety-text-qa', 
                'labsafety-scenario-decision', 'labsafety-scenario-qa', 
                'labsafety-decision-qa', 'labsafety-all', 'labsafety-text-qa-dpo']

# Add LabSafety task-specific model groups
labsafety_models = {
    'text_qa': ['labsafety-text-qa', 'labsafety-scenario-qa', 'labsafety-decision-qa', 'labsafety-all', 'labsafety-text-qa-dpo'],
    'scenario_hazards': ['labsafety-scenario', 'labsafety-scenario-decision', 'labsafety-scenario-qa', 'labsafety-all'],
    'decision_consequence': ['labsafety-decision', 'labsafety-scenario-decision', 'labsafety-decision-qa', 'labsafety-all']
}