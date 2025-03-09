import json
import numpy as np

# 对于模型列表中的 DA 结果（假设日志数据长度保持一致）
# 此处计算 Cronbach's Alpha，用于衡量一致性

score_array = np.zeros((13, 632))
models = ['llama3-instruct', 'vicuna', 'mistral', 'vicuna-13b', 'llama-3-70b', 
          'mistral-8x7b', 'galactica', 'claude3-haiku', 'claude3.5-sonnet', 
          'gemini-1.5-flash', 'gemini-1.5-pro', 'gpt-4o-mini', 'gpt-4o-2024-08-06']

for model_idx, model in enumerate(models):
    log_dir = '../../Logs/' + model + '/'
    mode = 'DA'
    n_shot = '0'
    file_path = log_dir + f'final_QA_{mode}_{n_shot}_result.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i, data_i in enumerate(data):
            if data_i['Correct Answer'] == data_i['LLM Answer']:
                score_array[model_idx][i] = 1

# 计算每项的样本方差
item_variances = np.var(score_array, axis=0, ddof=1)
# 计算每个模型的总分得分方差
total_scores = np.sum(score_array, axis=1)
total_variance = np.var(total_scores, ddof=1)
num_items = score_array.shape[1]
alpha = (num_items / (num_items - 1)) * (1 - (np.sum(item_variances) / total_variance))
print(f"Cronbach's Alpha: {alpha:.2f}")

# 同理可扩展至其他模型组（例如 _result_I.json），请根据实际情况调整数据维度 