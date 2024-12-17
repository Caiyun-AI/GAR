import ast
import json
import torch
import argparse
import numpy as np
import pandas as pd
import numpy as np  
from tqdm import tqdm  
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer


def load_model(cache_dir, model_name, proxies, device, only_load_tokenizer):
    if not only_load_tokenizer:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, proxies=proxies,
            local_files_only=False, low_cpu_mem_usage=True, attn_implementation="eager", use_safetensors=False).to(device, dtype=torch.float16)
        tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        return model, tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return tokenizer

def get_SNLI_features(model, tokenizer, texts, device, lhs):
    features = []
    labels = []
    for text in tqdm(texts):
        ids = tokenizer.encode_plus(text[0], return_tensors='pt').input_ids.to(device)
        output = model(ids, output_attentions = True)
        attentions = output.attentions
        tokens = tokenizer.tokenize(text[0])
        start = tokens.rindex('<0x0A>') if '<0x0A>' in tokens else 0 #'\n' is encoded as <0x0A>
        loc_start = tokens.index('▁So', start) + 1
        attention_weights = torch.cat(attentions)[:, :, loc_start:]
        max_attention_diff = attention_weights[:, :, :, 1:].max(-1).values - attention_weights[:, :, :, 0]
        max_value = max_attention_diff.max(-1).values.cpu().detach().numpy()
        features.append(max_value)
        labels.append(1 if text[1]=='Yes' else 0)
    Features = [np.stack(features, 2)[i,j] for i,j in lhs]
    result = Features+[labels]
    return result

def get_GoT_features(model, tokenizer, texts, device, lhs):
    features = []
    labels = []
    for text in tqdm(texts):
        ids = tokenizer.encode_plus(text, return_tensors='pt').input_ids.to(device)
        output = model(ids, output_attentions = True)
        attentions = output.attentions
        tokens = tokenizer.tokenize(text)
        start = tokens.rindex('<0x0A>') if '<0x0A>' in tokens else 0 #'\n' is encoded as <0x0A>
        loc_start = tokens.index('▁Is', start) + 1
        loc_end = tokens.index('▁Answer', start)
        attention_weights = torch.cat(attentions)[:, :, loc_start:loc_end]
        max_attention_diff = attention_weights[:, :, :, 1:].max(-1).values - attention_weights[:, :, :, 0]
        max_value = max_attention_diff.max(-1).values.cpu().detach().numpy()
        features.append(max_value)
        labels.append(1 if tokens[-1]=='▁Yes' else 0)
    Features = [np.stack(features, 2)[i,j] for i,j in lhs]
    result = Features+[labels]
    return result

def main(cache_dir, model_name, device, file_path, proxies, lhs, only_load_tokenizer):
    model, tokenizer = load_model(cache_dir, model_name, proxies, device, only_load_tokenizer)
    with torch.no_grad():
        with open(file_path, 'r', encoding='utf-8') as file:
            texts = json.load(file)
            if 'SNLI' in file_path:
                result = get_SNLI_features(model, tokenizer, texts, device, lhs)
            else:
                result = get_GoT_features(model, tokenizer, texts, device, lhs)
            data = pd.DataFrame(np.array(result).T, columns = lhs+['label'])
            file_address = file_path.replace(".jsonl",".csv")
            data.to_csv(f'./{file_address}', index=False)

    return f"Features has been written into {file_address}"

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--cache_dir", type=str, required=True, help="Your model path")
    parser.add_argument("--model_name", type=str, required=True, help="Your model name")
    parser.add_argument("--file_path", type=str, required=True, help="Your dataset path")
    parser.add_argument("--proxies", type=str, required=False, default=None, help="Your proxies")
    parser.add_argument("--device", type=int, required=False, default=0, help="Device to use (e.g., 0 for GPU0, 1 for GPU1, etc.)")
    parser.add_argument("--lhs", type=str, required=False, default='[(15,51),(14,1),(14,7),(14,0),(17,3),(18,11),(14,18),(14,46)]', help='Which (Layer, Head) attention weights do you want to use?')
    parser.add_argument("--only_load_tokenizer", type=bool, required=False, default=False)
    args = parser.parse_args()
    lhs = ast.literal_eval(args.lhs)
    main(cache_dir=args.cache_dir, model_name=args.model_name, file_path=args.file_path, 
         proxies=args.proxies, lhs=lhs, device=args.device, only_load_tokenizer=args.only_load_tokenizer)