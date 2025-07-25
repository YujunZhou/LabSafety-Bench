from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import openai
from openai import OpenAI
import base64
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import anthropic

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def analyze_image_with_prompt_llava_8B(model, processor, image_paths, prompt_texts, sys_prompt):
    prompts = [(
        f"<|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{prompt_texts[i]}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ) for i in range(len(prompt_texts))]
    generated_text_all = []
    batch_size = 16
    for idx in range(0, len(prompt_texts), batch_size):
        images = []
        for image_path in image_paths[idx:idx + batch_size]:
            image = Image.open(image_path)
            images.append(image)
        inputs = processor(
            [prompts[i] for i in range(idx, min(idx + batch_size, len(prompt_texts)))],
            images=images, return_tensors="pt", padding=True
        ).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated_texts = []
        for i in range(len(outputs)):
            generated_text = processor.decode(outputs[i][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            generated_texts.append(generated_text)
        generated_text_all += generated_texts
    return generated_text_all


def analyze_image_with_prompt_instructBlip(model, processor, image_paths, prompt_texts, sys_prompt):
    generated_texts_all = []
    batch_size = 1
    for idx in range(0, len(prompt_texts), batch_size):
        if image_paths is not None:
            images = [Image.open(image_paths[idx + i]) for i in range(batch_size)]
            inputs = processor(
                images=images,
                text=[sys_prompt + prompt_texts[idx]],
                return_tensors="pt", padding=True
            ).to("cuda")
        else:
            inputs = processor(
                text=[sys_prompt + prompt_texts[idx]],
                return_tensors="pt", padding=True
            ).to("cuda")
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
        generated = processor.batch_decode(outputs, skip_special_tokens=True)
        generated = [text.strip() for text in generated]
        generated_texts_all += generated
    return generated_texts_all


def analyze_image_with_prompt_Qwen(model, tokenizer, image_paths, prompt_texts, sys_prompt):
    responses = []
    for i in range(len(prompt_texts)):
        if image_paths is not None:
            query = tokenizer.from_list_format([
                {'image': image_paths[i]},
                {'text': sys_prompt + prompt_texts[i]},
            ])
        else:
            query = tokenizer.from_list_format([
                {'text': sys_prompt + prompt_texts[i]},
            ])
        response, history = model.chat(tokenizer, query=query, history=None)
        responses.append(response)
    return responses


def analyze_image_with_prompt_InternVL(model, tokenizer, image_paths, prompt_texts, sys_prompt):
    responses = []
    batch_size = 16
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    images = [load_image(img_path).to(torch.bfloat16).cuda() for img_path in image_paths]
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

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def analyze_image_with_prompt_gpt(image_path, prompt_text, model_name, sys_prompt):
    # Encode the image to base64
    base64_image = encode_image(image_path)
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

    # Return only the GPT model's responses
    return gpt_responses


def analyze_image_with_prompt_gemini(image_path, prompt_text, model_name, sys_prompt):
    # Encode the image to base64
    img = Image.open(image_path)
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

def analyze_image_with_prompt_claude(image_path, prompt_text, model_name, sys_prompt):
    # Encode the image to base64
    base64_image = encode_image(image_path)
    if 'jpg' in image_path:
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

def analyze_image_with_prompt_llama_3_2(model, processor, image_paths, prompt_texts, sys_prompt):
    images = [Image.open(image_paths[i]) for i in range(len(image_paths))]
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
    
