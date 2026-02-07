<div align="center">

<img src="/assets/logo.png" width="100%">

[![Website](https://img.shields.io/badge/Website-%F0%9F%8C%8D-blue?style=for-the-badge&logoWidth=40)](https://yujunzhou.github.io/LabSafetyBench.github.io/)
[![Paper](https://img.shields.io/badge/Paper-%F0%9F%8E%93-lightgrey?style=for-the-badge&logoWidth=40)](https://arxiv.org/abs/2410.14182)
[![Dataset](https://img.shields.io/badge/Dataset-%F0%9F%92%BE-green?style=for-the-badge&logoWidth=40)](https://huggingface.co/datasets/yujunzhou/LabSafety_Bench)

</div>

# LabSafety Bench: Benchmarking LLMs on Safety Issues in Scientific Labs

## 💡 Overview

Artificial Intelligence (AI) is revolutionizing scientific research, but its growing integration into laboratory environments brings critical safety challenges. As large language models (LLMs) and vision language models (VLMs) are increasingly used for procedural guidance and even autonomous experiment orchestration, there is a risk of an "illusion of understanding" where users may overestimate the reliability of these systems in safety-critical situations.

**LabSafety Bench** is a comprehensive evaluation framework designed to rigorously assess the trustworthiness of these models in laboratory settings. The benchmark includes two main evaluation components:

- **Multiple-Choice Questions (MCQs):**  
  A set of 765 questions derived from authoritative lab safety protocols, comprising 632 text-only questions and 133 multimodal questions.
  
- **Real-World Scenario Evaluations:**  
  A collection of 404 realistic laboratory scenarios that yield a total of 3128 open-ended questions, organized into:
  - **Hazards Identification Test:** Models identify all potential hazards in a given scenario.
  - **Consequence Identification Test:** Models predict the outcomes of executing specific hazardous actions.

Developed via expert-AI collaboration using sources such as OSHA, WHO, and established textbooks, LabSafety Bench ensures that every evaluation item is verified for clarity, accuracy, and practical relevance.

For more details, please visit our [project website](https://yujunzhou.github.io/LabSafetyBench.github.io/).

<div align="center">
<img src="/assets/Figure1_new.png" width="90%">
  
### LabSafety Bench Overview
</div>


## 🔧 Installation

Install the required Python packages by running:
```bash
pip install -r requirements.txt
```

### Additional Setup

For SFT (Supervised Fine-Tuning), please follow [@LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to install LLaMA-Factory.

For ChemCrow evaluation, please follow [@ChemCrow](https://github.com/ur-whitelab/chemcrow-public) and create a new environment for evaluation.

## 📖 Dataset Usage

### Data Downloading

The dataset is divided into five splits:
- **QA**: 632 text-only examples for standard evaluation.
- **QA_I**: 133 multimodal examples for standard evaluation.
- **sampledQA**: 80 text-only examples suitable for human evaluation, validation, or low-resource scenarios.
- **sampledQA_I**: 20 multimodal examples for similar use cases.
- **scenario**: 404 real-world scenarios combined with 3128 open-ended questions.

After installing [Huggingface Datasets](https://huggingface.co/docs/datasets/quickstart), download the dataset by running:
```python
from datasets import load_dataset

# Load MCQ configuration (default configuration)
MCQ_dataset = load_dataset("yujunzhou/LabSafety_Bench", name="MCQ")

# Or load a specific split from MCQ configuration
QA_split = load_dataset("yujunzhou/LabSafety_Bench", name="MCQ", split="QA")

# Load scenario configuration
scenario_dataset = load_dataset("yujunzhou/LabSafety_Bench", name="scenario", split="scenario")
```

### Data Format

#### MCQ Configuration ("MCQ")
Each sample in the MCQ configuration is a dictionary containing the following keys:
- **Question**: *string*  
  A multiple-choice question with four options.
- **Explanation**: *string*  
  A detailed explanation outlining why the correct answer is right and why the other options are not.
- **Correct Answer**: *string*  
  The correct option (one of 'A', 'B', 'C', or 'D').
- **Category**: *list of strings*  
  The lab safety category covered by the question.
- **Topic**: *string*  
  A brief descriptor identifying the main hazard or equipment involved.
- **Level**: *string*  
  “Easy” or “Hard”, indicating whether the question can be answered with high school-level knowledge.
- **Image Path**: *string*  
  The image file path for multimodal questions (None for text-only questions).
- **Decoded Image**: *Image*  
  The actual image for multimodal questions.

<div align="center">
<img src="/assets/Figure3.png" width="100%">
  
### Example Question Display
</div>


#### Scenario Configuration ("scenario")
Each sample in the scenario configuration is a dictionary containing the following keys:
- **Scenario**: *string*  
  A detailed description of the laboratory scenario.
- **LabSafety_Related_Issues**: *dict*  
  Contains:
  - **Most_Common_Hazards**: *list of strings*
  - **Improper_Operation_Issues**: *list of strings*
  - **Negative_Lab_Environment_Impacts**: *list of strings*
  - **Most_Likely_Safety_Incidents**: *list of strings*
- **Topic**: *string*  
  A brief descriptor identifying the main hazard or equipment involved.
- **SubCategory**: *string*  
  A subcategory label.
- **Decisions**: *list of dicts*  
  Each dictionary contains:
  - **Decision**: *string*
  - **Consequence**: *string*
- **Subject**: *string*  
  A Subject label.

## 📝 Evaluations

### 1. API Key Setup

Ensure that you have configured your OpenAI API key and any other required keys (e.g., for Claude or Gemini) in the `config.py` file.

### 2. Evaluations of Multiple-Choice Questions

LabSafety Bench supports evaluations for both text-only and multimodal tasks. Predefined models for text-only evaluations include, but are not limited to:
- **LLMs**: 'llama3-instruct-8b', 'vicuna-7b', 'mistral-7b', etc.
- **VLMs (for multimodal tasks)**: 'instructBlip-7B', 'Qwen-VL-Chat', 'InternVL2', etc.

Example commands for text-only MCQs evaluation on sampled MCQ dataset:
```sh
cd src/test

python text_QA.py \
--models gpt-4o-mini,o3-mini \
--mode CoT \
--n_shots 0 \
--sampled
```
For text-with-image MCQs evaluation:
```sh
python text_with_image_QA.py \
--model_name gpt-4o-mini \
--CoT \
--n_shots 0 \
```

Additional scripts such as `src/analysis/category_acc.py` and `src/analysis/level_acc.py` provide detailed breakdowns by safety category and difficulty level.

### 3. Evaluation of Real-World Scenario Tasks

The benchmark includes two additional real-world evaluation tasks:
- **Hazards Identification Test**: Assess the model's ability to comprehensively list potential hazards in realistic lab scenarios.
- **Consequence Identification Test**: Evaluate the model's capability to predict the outcomes of specific hazardous actions in a given scenario.

These tasks simulate dynamic and practical lab environments, addressing the critical need to ensure that AI systems are reliable when making safety-critical decisions.

Example commands for real-world scenario-based evaluation:

For scenario identification test:
```sh
python scenario_hazards.py \
--models gpt-4o-mini,o3-mini, llama3.3-70b \
--mode DA
```

For consequence identification test:
```sh
python decision_consequnce.py \
--models gpt-4o-mini,o3-mini, llama3.3-70b \
--mode CoT
```

For scenario hazards evaluation with set points:
```sh
python scenario_hazards_set_points.py \
--models gpt-4o-mini \
--mode DA \
--num_points 10
```

### 4. Evaluation of Additional Models

To evaluate open-weight models not included in the predefined list in "src/config.py", follow these steps:

1. **Configure Model Paths**: 
   First, add your model to `src/config.py` by setting the model name and path correspondence:
   ```python
   model_path_dicts = {
       # ... existing models ...
       "your-model-name": "/path/to/your/model",
       "another-model": "/path/to/another/model"
   }
   ```

2. **Run Evaluations**: 
   After configuring the model paths, run the evaluations from **Section 2 (Multiple-Choice Questions)** and **Section 3 (Real-World Scenario Tasks)** using your model names:
   ```sh
   # Example for MCQ evaluation
   python text_QA.py --models your-model-name --mode CoT --n_shots 0
   
   # Example for scenario evaluation  
   python scenario_hazards.py --models your-model-name --mode DA
   ```

3. **Advanced Customization**: 
   If needed, you can also modify the model loading and inference procedures in `src/utils` and adjust the corresponding evaluation scripts for specialized model architectures.

## 🔍 RAG (Retrieval-Augmented Generation) Enhancement

LabSafety Bench supports RAG enhancement to improve model performance by providing relevant safety knowledge context. The RAG system can retrieve pertinent information from safety documents to assist models in answering questions more accurately.

### RAG Setup

1. **Install Dependencies**:
   ```bash
   pip install llama-index
   ```

2. **Prepare Safety Documents**:
   Place your safety documents (PDF, TXT, etc.) in a directory. These should include:
   - Laboratory safety protocols
   - Chemical safety data sheets
   - Equipment operation manuals
   - Safety guidelines and regulations

3. **Build RAG Index**:
   ```bash
   cd src/rag
   python build_index.py \
   --input_dir /path/to/your/safety/documents \
   --persist_dir ./storage/safety_index \
   --embed_backend hf \
   --hf_model_name BAAI/bge-m3
   ```

### Using RAG in Evaluations

#### Option 1: Real-time RAG Retrieval

Enable RAG during evaluation by adding the `--use_rag` flag:

```sh
# MCQ evaluation with RAG
python text_QA.py \
--models gpt-4o-mini \
--mode CoT \
--use_rag \
--rag_persist_dir ./storage/safety_index \
--rag_top_k 2 \
--rag_hf_model BAAI/bge-m3

# Scenario evaluation with RAG
python scenario_hazards.py \
--models gpt-4o-mini \
--mode DA \
--use_rag \
--rag_persist_dir ./storage/safety_index \
--rag_top_k 2

# Decision consequence evaluation with RAG
python decision_consequence.py \
--models gpt-4o-mini \
--mode CoT \
--use_rag \
--rag_persist_dir ./storage/safety_index \
--rag_top_k 2
```

#### Option 2: Precomputed RAG Contexts

For faster evaluation, you can precompute RAG contexts and save them to JSON files:

```bash
# Generate contexts for text QA
python generate_contexts.py \
--persist_dir ./storage/safety_index \
--dataset text_qa \
--out ../../data/rag_contexts_text_qa.json

# Generate contexts for scenario variants
python generate_contexts.py \
--persist_dir ./storage/safety_index \
--dataset scenario \
--scenario_variant 1 \
--out ../../data/rag_contexts_scenario_v1.json

python generate_contexts.py \
--persist_dir ./storage/safety_index \
--dataset scenario \
--scenario_variant 2 \
--out ../../data/rag_contexts_scenario_v2.json

python generate_contexts.py \
--persist_dir ./storage/safety_index \
--dataset scenario \
--scenario_variant 3 \
--out ../../data/rag_contexts_scenario_v3.json

python generate_contexts.py \
--persist_dir ./storage/safety_index \
--dataset scenario \
--scenario_variant 4 \
--out ../../data/rag_contexts_scenario_v4.json

# Generate contexts for decision consequence
python generate_contexts.py \
--persist_dir ./storage/safety_index \
--dataset decision \
--out ../../data/rag_contexts_decision.json
```

Then use the precomputed contexts in evaluations:

```sh
# MCQ evaluation with precomputed contexts
python text_QA.py \
--models gpt-4o-mini \
--mode CoT \
--rag_context_path ../../data/rag_contexts_text_qa.json

# Scenario evaluation with precomputed contexts
python scenario_hazards.py \
--models gpt-4o-mini \
--mode DA \
--rag_context_v1 ../../data/rag_contexts_scenario_v1.json \
--rag_context_v2 ../../data/rag_contexts_scenario_v2.json \
--rag_context_v3 ../../data/rag_contexts_scenario_v3.json \
--rag_context_v4 ../../data/rag_contexts_scenario_v4.json

# Decision consequence evaluation with precomputed contexts
python decision_consequence.py \
--models gpt-4o-mini \
--mode CoT \
--rag_context_path ../../data/rag_contexts_decision.json
```

### RAG Tools

The `src/rag/` directory contains several utility scripts:

- **`build_index.py`**: Build or reload a LlamaIndex vector index from safety documents
- **`query_rag.py`**: Query an existing RAG index and print sources
- **`generate_contexts.py`**: Precompute RAG contexts for datasets and save to JSON
- **`evaluate_text_qa.py`**: Batch evaluate Text-QA using RAG query engine

### RAG Configuration Options

- **`--use_rag`**: Enable RAG enhancement (default: False)
- **`--rag_persist_dir`**: RAG index persistence directory
- **`--rag_top_k`**: Number of similar documents returned by RAG (default: 2)
- **`--rag_hf_model`**: HF embedding model used by RAG (default: "BAAI/bge-m3")
- **`--rag_context_path`**: Precomputed RAG context JSON file
- **`--rag_context_v1/v2/v3/v4`**: Precomputed context files for scenario variants

### Performance Considerations

- **Real-time RAG**: Slower but more flexible, retrieves relevant context for each query
- **Precomputed contexts**: Faster evaluation, contexts are pre-generated and cached
- **Embedding models**: Choose appropriate embedding models based on your document language and domain
- **Top-k selection**: Adjust the number of retrieved documents based on your needs (typically 2-5)

## 🚀 SFT Training and Evaluation

For all SFT settings, please first use LLaMA-Factory for training. The training datasets are located in `llamafactory_data`, which also includes `sft.yaml` as an SFT template. You only need to modify the dataset and output_dir to use it directly.

### Training with LLaMA-Factory

1. **Configure Dataset Registration**: 
   First, modify the `LLaMA-Factory/data/dataset_info.json` file to register our SFT datasets:

2. **Modify Training Configuration**: 
   Navigate to your LLaMA-Factory installation directory and modify the `sft.yaml` configuration file in `llamafactory_data` with your desired dataset and output directory.

3. **Run Training**:
   ```bash
   llamafactory-cli train sft.yaml
   ```

4. **Update Model Configuration**: 
   After training completion, modify `src/config.py` to add the trained model path and name correspondence:
   ```python
   model_path_dicts = {
       # ... existing models ...
       "labsafety-text-qa": "/path/to/your/fine-tuned/text-qa-model",
       "labsafety-scenario": "/path/to/your/fine-tuned/scenario-model", 
       "labsafety-decision": "/path/to/your/fine-tuned/decision-model"
   }
   ```

### Post-Training Evaluation

After training completion, use the following specialized SFT evaluation scripts for testing:

For MCQ evaluation with fine-tuned models:
```sh
cd src/test
python text_QA_sft.py \
--models labsafety-text-qa \
--mode CoT
```

For scenario hazards evaluation with fine-tuned models:
```sh
python scenario_hazards_sft.py \
--models labsafety-scenario \
--mode DA
```

For consequence identification with fine-tuned models:
```sh
python decision_consequence_sft.py \
--models labsafety-decision \
--mode CoT
```

These evaluation scripts are based on the existing `scenario_hazards_sft.py`, `decision_consequence_sft.py`, and `text_QA_sft.py` files, which have been specifically adapted for fine-tuned model evaluation with proper model loading and testing procedures.

### Further Analysis

For detailed analysis of results, you can directly use the following evaluation scripts:
- `src/analysis/category_acc.py` - Analyze accuracy by safety categories
- `src/analysis/level_acc.py` - Analyze accuracy by difficulty levels  
- `src/analysis/subject_acc.py` - Analyze accuracy by lab subjects


## ✅ Citation

If you use LabSafety Bench in your research, please cite our work:
```
@article{zhou2026benchmarking,
  title={Benchmarking large language models on safety risks in scientific laboratories},
  author={Zhou, Yujun and Yang, Jingdong and Huang, Yue and Guo, Kehan and Emory, Zoe and Ghosh, Bikram and Bedar, Amita and Shekar, Sujay and Liang, Zhenwen and Chen, Pin-Yu and others},
  journal={Nature Machine Intelligence},
  pages={1--12},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```
