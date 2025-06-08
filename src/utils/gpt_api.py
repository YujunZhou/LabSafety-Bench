import openai
from pydantic import BaseModel
from typing import List

# Define Pydantic models
class Json_QA(BaseModel):
    Question: str
    Explanation: str
    Correct_Answer: str

class Json_Decision(BaseModel):
    class DecisionDetail(BaseModel):
        Decision: str
        Consequence: str
    Decisions: List[DecisionDetail]

# OpenAI API calling class
class OpenAIModel:
    def __init__(self, model_name, max_tokens=None, system_prompt='You are a helpful assistant.', temperature=0.0, top_p=0.9, **kwargs):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p

    def _compose_messages(self, content, image=None):
        if image is None:
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}
            ]
        else:
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": content},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                ]}
            ]

    def _call_api_create(self, content):
        client = openai.OpenAI()
        return client.chat.completions.create(
            model=self.model_name,
            messages=self._compose_messages(content),
            max_tokens=self.max_tokens,
            temperature=self.temperature
        ).choices[0].message.content

    def _call_api_beta(self, content, response_format, beta_model="gpt-4o-2024-11-20", image=None):
        client = openai.OpenAI()
        return client.beta.chat.completions.parse(
            model=beta_model,
            messages=self._compose_messages(content, image=image),
            response_format=response_format
        ).choices[0].message.content

    def prompt(self, x):
        return self._call_api_create(x)

def get_embedding(texts):
    # No need to set api_key, directly use environment variables
    client = openai.OpenAI()
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [response.data[i].embedding for i in range(len(texts))] 