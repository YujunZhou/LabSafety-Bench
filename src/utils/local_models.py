import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM

def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    # Check GPU availability
    if 'cuda' in device and not torch.cuda.is_available():
        print("Warning: GPU requested but CUDA not available, falling back to CPU")
        device = 'cpu'
    else:
        print(f"Using device: {device}")
        if 'cuda' in device:
            print(f"GPU info: {torch.cuda.get_device_name(0)}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    if 'galactica' not in model_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if 'cuda' in device else torch.float32,
            trust_remote_code=True,
            **kwargs
        ).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            padding_side='left'
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
        
        tokenizer.padding_side = 'left'
    elif 'galactica' in model_path:
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b")
        model = OPTForCausalLM.from_pretrained("facebook/galactica-6.7b", device_map="auto", **kwargs)
    else:
        model = None
        tokenizer = None

    return model, tokenizer

def adaptive_generate_batch(input_ids_tensor_batch, attn_mask_batch, model, tokenizer, current_bs, max_new_tokens, random_sample):
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
    try:
        return model.generate(
            input_ids_tensor_batch,
            attention_mask=attn_mask_batch,
            generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if current_bs == 1:
            raise RuntimeError("Out of memory for single sample generation")
        new_bs = max(1, current_bs // 2)
        outputs_list = []
        total = input_ids_tensor_batch.size(0)
        for i in range(0, total, new_bs):
            sub_input = input_ids_tensor_batch[i:i + new_bs]
            sub_mask = attn_mask_batch[i:i + new_bs]
            outputs_list.append(adaptive_generate_batch(sub_input, sub_mask, model, tokenizer, new_bs, max_new_tokens, random_sample))
        return torch.cat(outputs_list, dim=0)

def _llm_generate_core(inputs, conv_template, model, tokenizer, batch_size, max_new_tokens, random_sample):
    import torch
    input_ids_list = []
    
    # Add input length validation
    max_model_length = model.config.max_position_embeddings
    for i in range(len(inputs)):
        conv_template.append_message(conv_template.roles[0], inputs[i])
        conv_template.append_message(conv_template.roles[1], None)
        prompt = conv_template.get_prompt()
        # Add truncation handling
        encoding = tokenizer(prompt, truncation=True, max_length=max_model_length - max_new_tokens)
        toks = encoding.input_ids
        input_ids = torch.tensor(toks).to(model.device)
        input_ids_list.append(input_ids)
        conv_template.messages = []

    pad_tok = tokenizer.pad_token_id
    if pad_tok is None:  # Ensure pad_token exists
        pad_tok = tokenizer.eos_token_id
    
    # Change to left padding (more suitable for most LLMs)
    max_input_length = max(ids.size(0) for ids in input_ids_list)
    padded_input_ids_list = [
        torch.cat([
            torch.full((max_input_length - ids.size(0),), pad_tok, device=model.device),
            ids
        ], dim=0) for ids in input_ids_list
    ]
    
    input_ids_tensor = torch.stack(padded_input_ids_list, dim=0)
    attn_mask = (input_ids_tensor != pad_tok).type(input_ids_tensor.dtype)
    
    # Add CUDA memory cleanup
    torch.cuda.empty_cache()
    
    output_ids_new = []
    for i in range(0, len(input_ids_tensor), batch_size):
        input_ids_tensor_batch = input_ids_tensor[i:i + batch_size]
        attn_mask_batch = attn_mask[i:i + batch_size]
        
        # Add safety check
        if input_ids_tensor_batch.size(1) > max_model_length:
            raise ValueError(f"Input length {input_ids_tensor_batch.size(1)} exceeds model max length {max_model_length}")
            
        output_ids_batch = adaptive_generate_batch(input_ids_tensor_batch, attn_mask_batch, 
                                                 model, tokenizer, batch_size, max_new_tokens, random_sample)
        for output_ids in output_ids_batch:
            output_ids_new.append(output_ids[input_ids_tensor_batch.size(1):])  # Fix slicing method
            
    decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in output_ids_new]
    return decoded

def llm_generate_QA_answer(inputs, conv_template, model, tokenizer, batch_size=6, random_sample=False, max_new_tokens=512):
    return _llm_generate_core(inputs, conv_template, model, tokenizer, batch_size, max_new_tokens, random_sample) 