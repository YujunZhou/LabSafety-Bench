import openai
from openai import OpenAI
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, LlamaTokenizer, LlamaForCausalLM
import torch
import base64
from PIL import Image
from pydantic import BaseModel
from typing import List
import replicate
import google.generativeai as genai
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import anthropic
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import io


os.environ["REPLICATE_API_TOKEN"] = ''
os.environ['OPENAI_API_KEY'] = ''
os.environ['GOOGLE_API_KEY'] = ''
os.environ['ANTHROPIC_API_KEY'] = ''

class OpenAIModel:
    def __init__(self, model_name, max_tokens=None, system_prompt='You are a helpful assistant.', temperature=0.0, top_p=0.9, **kwargs):

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def prompt(self, x):
        client = OpenAI()
        return client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": x}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        ).choices[0].message.content

    def evolve_QA_Json(self, x):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI()
        return client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": x}
            ],
            response_format=Json_QA_Harder
        ).choices[0].message.content

    def evolve_prompt_I(self, x, image):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI()
        return client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [
                    {
                      "type": "text",
                      "text": x
                    },
                    {
                      "type": "image_url",
                      "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                      }
                    }]
                }
            ],
            response_format=Json_QA_Harder
        ).choices[0].message.content

# Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model_paths = {
    "mistral-8x7b": "mistralai/mixtral-8x7b-instruct-v0.1",
    'llama-3-70b': "meta/meta-llama-3-70b-instruct",
}

def text_completion_open_source_models(prompt, model, system_prompt='You are a helpful assistant.', temperature=0.1, max_tokens=512):
    input = input_completion(prompt, model, system_prompt, temperature, max_tokens)

    output = ''
    for event in replicate.stream(
            model_paths[model],
            input=input
    ):
        output += str(event)
    return output

def input_completion(prompt, model, system_prompt='You are a helpful assistant.', temperature=0.6, max_tokens=1024):
    if model == "mistral-8x7b":
        input = {
            "top_k": 50,
            "top_p": 0.9,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "prompt_template": "<s>[INST] {prompt} [/INST] "
        }
    elif model == 'llama-3-70b':
        input = {
            "top_p": 0.9,
            "prompt": prompt,
            "min_tokens": 0,
            "temperature": temperature,
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 1.15
        }
    else:
        input = {
            "prompt": prompt
        }
    return input

class Json_QA_Harder(BaseModel):
    Correct_Answer: str
    Assessment_of_Incorrect_Options_Difficulty: str
    Replacement_of_Easiest_to_Judge_Options_with_Relevant_Knowledge_Points: str
    Modified_Question: str
    Explanation: str


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    if 'galactica' not in model_path and 'darwin' not in model_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs
        ).to(device).eval()

        tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False,
            padding_side='left',
        )
        tokenizer.pad_token = tokenizer.unk_token
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    elif 'galactica' in model_path:
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b")
        model = OPTForCausalLM.from_pretrained("facebook/galactica-6.7b", device_map="auto", **kwargs)
    else:
        model = None
        tokenizer = None

    return model, tokenizer


def llm_generate_QA_answer(inputs, conv_template, model, tokenizer, batch_size=6, random_sample=False, max_new_tokens=512):
    input_ids_list = []
    for i in range(len(inputs)):
        conv_template.append_message(conv_template.roles[0], inputs[i])
        conv_template.append_message(conv_template.roles[1], None)
        prompt = conv_template.get_prompt()
        encoding = tokenizer(prompt)
        toks = encoding.input_ids
        input_ids = torch.tensor(toks).to(model.device)
        input_ids_list.append(input_ids)
        conv_template.messages = []
    pad_tok = tokenizer.pad_token_id
    max_input_length = max([ids.size(0) for ids in input_ids_list])
    # Pad each input_ids tensor to the maximum length
    padded_input_ids_list = []
    for ids in input_ids_list:
        pad_length = max_input_length - ids.size(0)
        padded_ids = torch.cat([torch.full((pad_length,), pad_tok, device=model.device), ids], dim=0)
        padded_input_ids_list.append(padded_ids)
    input_ids_tensor = torch.stack(padded_input_ids_list, dim=0)
    attn_mask = (input_ids_tensor != pad_tok).type(input_ids_tensor.dtype)
    generation_config = model.generation_config
    generation_config.max_new_tokens = max_new_tokens
    if random_sample:
        generation_config.do_sample = True
        generation_config.temperature = 0.6
        generation_config.top_p = 0.9
    else:
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
    flag = False
    while not flag:
        try:
            output_ids_new = []
            for i in range(0, len(input_ids_tensor), batch_size):
                input_ids_tensor_batch = input_ids_tensor[i:i + batch_size]
                attn_mask_batch = attn_mask[i:i + batch_size]
                output_ids_batch = model.generate(input_ids_tensor_batch,
                                                  attention_mask=attn_mask_batch,
                                                  generation_config=generation_config,
                                                  pad_token_id=tokenizer.pad_token_id)

                for j in range(len(output_ids_batch)):
                    output_ids_new.append(output_ids_batch[j][max_input_length:])
            flag = True
        # except cuda out of memory error
        except torch.cuda.OutOfMemoryError:
            batch_size = batch_size // 2
    analyses = [tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids_new]

    return analyses

def analyze_image_with_prompt_llava_8B(model, processor, decoded_images, prompt_texts, sys_prompt):
    prompts = [(f"<|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{prompt_texts[i]}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n") for i in range(len(prompt_texts))]
    generated_text_all = []



    batch_size = 16
    for idx in range(0, len(prompt_texts), batch_size):
        images = []
        for decoded_image in decoded_images[idx:idx + batch_size]:
            images.append(decoded_image)
        inputs = processor([prompts[i] for i in range(idx, min(idx + batch_size, len(prompt_texts)))],
                           images=images, return_tensors="pt", padding=True).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated_texts = []
        for i in range(len(outputs)):
            generated_text = processor.decode(outputs[i][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            # generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
            generated_texts.append(generated_text)
        generated_text_all = generated_text_all + generated_texts

    return generated_text_all


def analyze_image_with_prompt_instructBlip(model, processor, decoded_images, prompt_texts, sys_prompt):
    generated_texts_all = []
    batch_size = 1
    for idx in range(0, len(prompt_texts), batch_size):
        if decoded_images is not None:
            images = []
            for decoded_image in decoded_images[idx:idx + batch_size]:
                images.append(decoded_image)
            prompts = prompt_texts
            inputs = processor(images=images, text=[sys_prompt + prompts[i] for i in range(idx, min(idx + batch_size, len(prompt_texts)))], return_tensors="pt", padding=True).to("cuda")
        else:
            prompts = prompt_texts
            inputs = processor(text=[sys_prompt + prompts[i] for i in range(len(prompts))], return_tensors="pt", padding=True).to("cuda")

        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=512,
            min_length=1,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=0.1,
        )
        generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = [text.strip() for text in generated_texts]
        generated_texts_all = generated_texts_all + generated_texts
    return generated_texts_all


def analyze_image_with_prompt_Qwen(model, tokenizer, decoded_images, prompt_texts, sys_prompt):
    responses = []
    for i in range(len(prompt_texts)):
        if decoded_images is not None:
            buffered = io.BytesIO()
            decoded_images[i].save(buffered, format=decoded_images[i].format)
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            query = tokenizer.from_list_format([
                {'image': img_base64},
                {'text': sys_prompt + prompt_texts[i]},
            ])
        else:
            query = tokenizer.from_list_format([
                {'text': sys_prompt + prompt_texts[i]},
            ])
        response, history = model.chat(tokenizer, query=query, history=None)
        responses.append(response)
    return responses


def analyze_image_with_prompt_InternVL(model, tokenizer, decoded_images, prompt_texts, sys_prompt):
    responses = []
    batch_size = 16
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    images = []
    for decoded_image in decoded_images:
        image = load_image(decoded_image).to(torch.bfloat16).cuda()
        images.append(image)
    for i in range(len(prompt_texts)):
        question = f'<image>\n{sys_prompt + prompt_texts[i]}'
        response = model.chat(tokenizer, images[i], question, generation_config)
        responses.append(response)
    return responses


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(decoded_image, input_size=448, max_num=12):
    image = decoded_image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def analyze_image_with_prompt_gpt(decoded_image, prompt_text, model_name, sys_prompt):
    # Encode the image to base64
    buffered = io.BytesIO()
    decoded_image.save(buffered, format=decoded_image.format)
    img_bytes = buffered.getvalue()
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    client = OpenAI()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": sys_prompt + "\n" + prompt_text
                },
            ],
        }
    ]

    messages[0]['content'].append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        }
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=512,
        temperature=0
    )

    gpt_responses = response.choices[0].message.content
    return gpt_responses


def analyze_image_with_prompt_gemini(img, prompt_text, model_name, sys_prompt):
    # Encode the image to base64
    model = genai.GenerativeModel(model_name)

    response = model.generate_content([
                                          sys_prompt + "\n" + prompt_text,
                                          img], stream=True, safety_settings={
                                                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE
                                            })
    response.resolve()
    return response.text

def analyze_image_with_prompt_claude(decoded_image, prompt_text, model_name, sys_prompt):
    # Encode the image to base64
    buffered = io.BytesIO()
    decoded_image.save(buffered, format=decoded_image.format)
    img_bytes = buffered.getvalue()
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    if 'JPEG' in decoded_image.format:
        image_media_type = 'image/jpeg'
    else:
        image_media_type = 'image/png'


    analysis = anthropic.Anthropic().messages.create(
        model=model_name,
        max_tokens=1024,
        system=sys_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": base64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ],
            }
        ],
    ).content[0].text

    return analysis

def analyze_image_with_prompt_llama_3_2(model, processor, images, prompt_texts, sys_prompt):
    analyses = []
    for i in range(len(prompt_texts)):
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": sys_prompt + '\n' +  prompt_texts[i]}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images[i], input_text, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=512)
        analysis = processor.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        analyses.append(analysis)
        print(analysis)
    return analyses
def analyze_image_with_prompt(model_name, image_path, prompt_text, sys_prompt, model=None, processor=None):
    if 'gpt-4' in model_name:
        return analyze_image_with_prompt_gpt(image_path, prompt_text, model_name, sys_prompt=sys_prompt)
    elif 'gemini' in model_name:
        return analyze_image_with_prompt_gemini(image_path, prompt_text, model_name, sys_prompt=sys_prompt)
    elif 'claude' in model_name:
        return analyze_image_with_prompt_claude(image_path, prompt_text, model_name, sys_prompt=sys_prompt)
    elif 'InternVL' in model_name:
        return analyze_image_with_prompt_InternVL(model, processor, image_path, prompt_text, sys_prompt=sys_prompt)
    elif "llava" in model_name:
        return analyze_image_with_prompt_llava_8B(model, processor, image_path, prompt_text, sys_prompt=sys_prompt)
    elif "instructBlip" in model_name:
        return analyze_image_with_prompt_instructBlip(model, processor, image_path, prompt_text, sys_prompt=sys_prompt)
    elif 'Qwen' in model_name:
        return analyze_image_with_prompt_Qwen(model, processor, image_path, prompt_text, sys_prompt=sys_prompt)
    elif 'llama3_2' in model_name:
        return analyze_image_with_prompt_llama_3_2(model, processor, image_path, prompt_text, sys_prompt=sys_prompt)
    else:
        raise ValueError("Invalid model name. Choose 'gpt' or 'claude'.")