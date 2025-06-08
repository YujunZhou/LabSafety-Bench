import json
import numpy as np
from collections import OrderedDict
import csv

# Define four subjects
subjects = ['Biology', 'Chemistry', 'Physics', 'General']

model_dict = {}
number_dict = {}

for model in ['gpt-4o-mini', 'gpt-4o-2024-08-06', 'o3-mini', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash',
              'claude3-haiku', 'claude3.5-sonnet', 'deepseek-r1','llama3-instruct', 'llama-3-70b', 'mistral',
              'mistral-8x7b', 'vicuna', 'vicuna-13b']:
    log_dir = '../../Logs/' + model + '/'
    mode = 'DA'
    n_shot = '0'
    file_path = log_dir + f'final_QA_{mode}_{n_shot}_result.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if model not in model_dict:
            model_dict[model] = [0] * 4  # 4 subjects
            number_dict[model] = [0] * 4  # 4 subjects
        
        for data_i in data:
            category = data_i['Category']
            
            # Determine which subjects the question belongs to
            is_biology = 'biological hazards' in category
            is_chemistry = 'chemical hazards' in category
            is_physics = any(item in category for item in ['physical hazards', 'electricity safety', 'radiation hazards'])
            is_general = not (is_biology or is_chemistry or is_physics)
            
            # Count the number of questions and accuracy for each subject
            for idx, is_subject in enumerate([is_biology, is_chemistry, is_physics, is_general]):
                if is_subject:
                    number_dict[model][idx] += 1
                    if data_i['Correct Answer'] == data_i['LLM Answer']:
                        model_dict[model][idx] += 1

# Calculate the accuracy percentage for each subject
for model in model_dict:
    model_dict[model] = [round(model_dict[model][i] / number_dict[model][i], 4) * 100 
                         if number_dict[model][i] != 0 else 0 for i in range(4)]

# Define model order
order = ['gpt-4o-mini', 'gpt-4o-2024-08-06', 'o3-mini', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash',
         'claude3-haiku', 'claude3.5-sonnet', 'deepseek-r1','llama3-instruct', 'llama-3-70b', 'mistral',
         'mistral-8x7b', 'vicuna', 'vicuna-13b']

model_dict = OrderedDict((key, model_dict[key]) for key in order)

# Save results to csv file
with open('../../subject_acc.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Model', 'Biology', 'Chemistry', 'Physics', 'General'])
    for model in model_dict:
        writer.writerow([model] + model_dict[model])

# Print the total number of questions for each subject
print("Subject question statistics (based on first model's data):")
first_model = list(number_dict.keys())[0]
for i, subject in enumerate(subjects):
    print(f"{subject}: {number_dict[first_model][i]} questions") 