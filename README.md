<div align="center">

<img src="/assets/logo.png" width="100%">


[![Website](https://img.shields.io/badge/Website-%F0%9F%8C%8D-blue?style=for-the-badge&logoWidth=40)](https://yujunzhou.github.io/LabSafetyBench.github.io/)
[![Paper](https://img.shields.io/badge/Paper-%F0%9F%8E%93-lightgrey?style=for-the-badge&logoWidth=40)]()
[![Dataset](https://img.shields.io/badge/Dataset-%F0%9F%92%BE-green?style=for-the-badge&logoWidth=40)](https://huggingface.co/datasets/yujunzhou/LabSafety_Bench)


</div>

The official repository of "LabSafety Bench: Benchmarking LLMs on Safety Issues in Scientific Labs" 

## 💡Overview




We propose LabSafety Bench, a specialized evaluation framework designed to assess the reliability and safety awareness of LLMs in laboratory environments. First, we propose a new taxonomy for lab safety, aligned with US Occupational Safety and Health Administration (OSHA) protocols. Second, we curate a set of 765 multiple-choice questions guided by this taxonomy to ensure comprehensive coverage of safety concerns across various domains. Of these, 632 are text-only questions, while 133 are text-with-image questions.

<div align="center">
<img src="/assets/Taxonomy.png" width="100%">
  
###  **Our proposed new taxonomy for lab safety.**
  
<img src="/assets/Figure3.png" width="100%">
  
### **Some examples of LabSafety Bench.**

</div>



For more details about TrustLLM, please refer to [project website](https://yujunzhou.github.io/LabSafetyBench.github.io/).


## 🔧 Instrallation

 **Install Required Packages**  
   Install the necessary Python packages by running:
   ```bash
   pip install -r requirements.txt
   ```
## 📖 Dataset Usage
### Data Downloading

The data examples are divided into four splits: 'QA', 'QA_I', 'sampledQA', 'sampledQA_I'

- **QA**: 632 text-only examples for standard evaluation.
- **QA_I**: 133 multimodal questions for standard evaluation.
- **sampledQA**: 80 text-only examples used for human evaluation, validation, or for those with limited computing resources.
- **sampledQA_I**: 20 multimodal examples used for human evaluation, validation, or for those with limited computing resources.

You can download this dataset by the following command (make sure that you have installed [Huggingface Datasets](https://huggingface.co/docs/datasets/quickstart)):

```python
from datasets import load_dataset
# Load all the splits
dataset = load_dataset("yujunzhou/LabSafety_Bench")
# Load one of the splits, split can choose one of the following: 'QA', 'QA_I', 'sampledQA', 'sampledQA_I'
QA_dataset = load_dataset('yujunzhou/LabSafety_Bench', split='QA')
```

### Data Format
Each data item in the dataset is a dict with the following keys: "Question", "Correct Answer", "Explanation", "Category", "Topic", "Level", "Image Path" and "Decoded Image", as shown below:
```
{
    "Question": [str] A multiple-choice question with four options,
    "Explanation": [str] An explanation of the question why the correct answer is correct and why the wrong answers are wrong,
    "Correct Answer": [char] One option from 'A', 'B', 'C' or 'D',
    "Category": [list] The category that the question covers about lab safety,
    "Topic": [str] A brief word or phrase about the main hazardous substance or equipment involved in the question,
    "Level": [str] "Easy" or "Hard" indicating whether this question can be answered within high school knowledge,
    "Image Path": [str] Image path for the multimodal questions. For text-only questions, it is None,
    "Decoded Image": [Image] The shown image for multimodal questions
}
```

## 📝 Evaluations
1. **API Key Setup**
   - Since we use GPT-4o-mini as the answer extractor, you need to first set **OpenAI API** in "utils.py"
   - Add API keys for the other models you need to evaluate, like **Claude** and **Gemini** API in `utils.py`.
2. **Evaluations of the predefined models**
   We predefine some models in the evaluation, you can directly evaluate these models on the datasets without any modification.
   
   The predefined models for text-only questions are: ['llama3-instruct-8b', 'vicuna-7b', 'mistral-7b', 'vicuna-13b', 'llama-3-70b', 'mistral-8x7b', 'galactica-6.7b', 'claude3-haiku', 'claude3.5-sonnet', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gpt-4o-mini', 'gpt-4o-2024-08-06']
   
   The predefined models for multimodal questions are: ['instructBlip-7B', 'Qwen-VL-Chat', 'InternVL2', 'llama3_2', 'claude3-haiku', 'claude3.5-sonnet', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gpt-4o-mini', 'gpt-4o-2024-08-06']

   By providing one model name specified above, you can evaluate the model on the LabSafety Bench. You can set the number of shots and decide whether to use CoT or the sampled dataset. For "text.py", you can also set whether to use hints in the evaluation. Here is an example.
   ```sh
cd src

python test.py \
--model_name gpt-4o-mini \
--CoT
--use_hint
--n_shots 0
--sampled
```

```sh
python testV.py \
--model_name gpt-4o-mini \
--CoT
--n_shots 0
--sampled
```
After that, you can run "category_acc.py" or "level_acc.py" to get the accuracy of specified models in different categories or levels.

3. **Evaluation of other models**
   For other models, you need to specify how to load the model and tokenizer and how to make inferences in "utils.py" and then change "test.py" or "testV.py" accordingly.


