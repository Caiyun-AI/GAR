import argparse
import ast
import sys
import os
from collections import defaultdict, Counter
try: from collections import Iterable
except Exception: from collections.abc import Iterable  # for Python >= 3.10
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F 

from GAR_utils import _cxt2str
from GAR_utils import *
from data import *
from itertools import product
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer


tasks_simple = [
    (lambda: [TreeSet(countries_of_cities).use(['equal']), TreeSet(countries_of_cities).use(['child'])], rlr_gen,
     lambda *args, **kwargs: '',lambda q, _: f"{q} is",# lambda q, _: f"{q} is",
    ),
    (lambda: [TreeSet(countries_of_landmarks).use(['equal']), TreeSet(countries_of_landmarks).use(['child'])], rlr_gen,
     lambda *args, **kwargs: '',lambda q, _: f"{q} is",# lambda q, _: f"{q} is",
    ),
    (lambda: [TreeSet(kinds_of_things).use(['equal']), TreeSet(kinds_of_things).use(['child'])], rlr_gen,
     lambda *args, **kwargs: '', lambda q, _: f"{q} is",
    ),
    (lambda: [TreeSet(genders_of_persons).use(['equal']), TreeSet(genders_of_persons).use(['child'])], rlr_gen,
     lambda *args, **kwargs: '', lambda q, _: f"{q} is usually a",
    ),
    (lambda: [TreeSet(occupations_Of_Persons).use(['equal']), TreeSet(occupations_Of_Persons).use(['child'])], rlr_gen,
     lambda *args, **kwargs: '', lambda q, _: f"{q} is",
    ),
    (lambda: [TreeSet(capabilities_of_things).use(['equal']), TreeSet(capabilities_of_things).use(['child'])], rlr_gen,
     lambda *args, **kwargs: '', lambda q, _: f"{q} is",
    ),
    (lambda: [TreeSet(person_adjs).use(['equal']), TreeSet(capabilities_of_things).use(['similar'])], rlr_gen,
     lambda *args, **kwargs: '', lambda q, _: f"{q} is",
    ),
]
ins = 'So '
tasks = [
    (lambda: [TreeSet(genders_of_persons).use(['equal', 'child']), TreeSet(countries_of_cities).use(['equal', 'child'])], partial(rlr_gen, dict_candidates=True),
     '', partial(_cxt2str, item2str=lambda i, _: [f"{i[0]} likes {i[1]}.", f"{i[1]} attracts {the_(i[0], uppercase=False)}."]), lambda q, _: f'{ins}{(q)} wants to go to',
    ),
    (lambda: [TreeSet(genders_of_persons).use(['equal', 'child']), TreeSet(countries_of_landmarks).use(['equal', 'child'])], partial(rlr_gen, dict_candidates=True),
     '', partial(_cxt2str, item2str=lambda i, _: [f"{i[0]} likes {i[1]}.", f"{i[1]} attracts {the_(i[0], uppercase=False)}."]), lambda q, _: f'{ins}{(q)} wants to go to',
    ),
    (lambda: [TreeSet(genders_of_persons).use(['equal', 'child']), TreeSet(kinds_of_things).use(['equal', 'child'])], partial(rlr_gen, dict_candidates=True),
    ('', None), partial(_cxt2str, item2str=lambda i, _: [f"{i[0]} has {(i[1])}.", f"{(the_(i[1], uppercase=True))} is {i[0]}'s."]), lambda q, _: f"{ins}{(q)} owns",
    ),
    (lambda: [TreeSet(genders_of_persons).use(['equal', 'child']), TreeSet(capabilities_of_things).use(['child'])], partial(rlr_gen, dict_candidates=True),
     '', partial(_cxt2str, item2str=lambda i, _: [f"{i[0]} has {(i[1])}.",f"{(the_(i[1], uppercase=True))} is {i[0]}'s."]), lambda q, _: f"{ins}{(q)} owns a thing used for"
    ),
    (lambda: [TreeSet(occupations_Of_Persons).use(['child']), TreeSet(countries_of_cities).use(['equal','child'])], rlr_gen,
    '', partial(_cxt2str, item2str=lambda i, _: [f"{the_(i[0], uppercase=False)} likes {i[1]}.", f"{i[1]} attracts {the_(i[0])}."]),
     lambda q, _: f"So {q} wants to go to",
    ),
    (lambda: [TreeSet(occupations_Of_Persons).use(['child']), TreeSet(countries_of_landmarks).use(['equal','child'])], rlr_gen,
    '', partial(_cxt2str, item2str=lambda i, _: [f"{i[0]} likes {i[1]}.", f"{i[1]} attracts {the_(i[0])}."]),
     lambda q, _: f"So {q} wants to go to",
    ),
    (lambda: [TreeSet(occupations_Of_Persons).use(['child']), TreeSet(kinds_of_things).use(['equal','child'])], rlr_gen,
    ('', None), partial(_cxt2str, item2str=lambda i, _: [f"{i[0]} has {i[1]}.", f"{(the_(i[1], uppercase=True))} is {i[0]}'s."]),
     lambda q, _: f"So {q} owns",
    ),
    (lambda: [TreeSet(occupations_Of_Persons).use(['child']), TreeSet(capabilities_of_things).use(['child'])], rlr_gen,
    '', partial(_cxt2str, item2str=lambda i, _: [f"{i[0]} has {i[1]}.", f"{(the_(i[1], uppercase=True))} is {i[0]}'s."]),
     lambda q, _: f"So {q} has a thing used for",
    ),
    (lambda: [TreeSet(genders_of_persons).use(['equal', 'child']), SymSet(person_adjs).use(['similar'])], rlr_gen,
     '', partial(_cxt2str,item2str=lambda i, _: [f"{i[0]} is {i[1]}.", f"{capitalize(i[1])} {i[0]}."]), lambda q, _: f"{ins}{(q)} is",
    ),
    (lambda: [TreeSet(occupations_Of_Persons).use(['child']), SymSet(person_adjs).use(['similar'])], rlr_gen,
     '', partial(_cxt2str,item2str=lambda i, _: [f"{i[0]} is {i[1]}.", f"{capitalize(i[1])} {i[0]}."]), lambda q, _: f"In other words, {q} is",
    ),  
]

def load_model(cache_dir, model_name, proxies, device, only_load_tokenizer):
    if not only_load_tokenizer:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, proxies=proxies,
            local_files_only=False, low_cpu_mem_usage=True, attn_implementation="eager", use_safetensors=False).to(device, dtype=torch.float16)
        tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        return model, tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return tokenizer

def generate(task, nrows=8, cxt_len=3, rev_item2str=False, abstract=0, counter_paired=False,
            tokenizer=None, max_length=512, plot=True, verbose=True):
    vocab_fn, gen_fn, cxt2str, query2str, *a = task
    position_relevant = isinstance(gen_fn, partial) and \
        'cxt_sample_fn' in gen_fn.keywords and 'query' in gen_fn.keywords and \
        gen_fn.keywords['cxt_sample_fn'].__name__ == 'enumerate_sample'

    # ans_counts = [('a', nrows)]; ind_counts = [(0, 9), (1, 1)]
    i = 0
    conditions = [True, ]
    while (i == 0 or nrows >= 4) and any(conditions):
        vocabs, examples = make_examples(task, nrows=nrows, cxt_len=cxt_len, counter_paired=counter_paired)
        if counter_paired: break  # avoid balance checking for g2c tasks
        cxt, query, candidates, (tgt, *_, ans0, ans), *cls = examples[0]
        if len(cls) > 0: break  # avoid balance checking for g2c tasks
        ans_counts = Counter([ans for cxt, query, cands, (*_, ans), *cls in examples]).most_common()
        answer_indices = [get_answer_index(e) for e in examples]
        ind_counts = Counter(answer_indices).most_common()
        conditions = [
            not position_relevant and len(ind_counts) > 1 and (len(ind_counts) < cxt_len 
                                    or ind_counts[0][1] > ind_counts[-1][1] * 3),
            len(ans_counts) == 1,
            len(ans_counts) > 2 and ans_counts[0][1] > max(2, nrows / 3),
            len(ans_counts) == 2 and ans_counts[0][1] > ans_counts[1][1] * 2,
        ]
        i += 1
        assert i < 60, str(conditions) + '\n'.join(f'{e[0]}\t{e[1]}\t{e[3]}' for e in examples[:]) + \
            '\n' + str(ind_counts) + '\n' + str(ans_counts) 
    if i > 10: print('In generate: i =', i, task2str(task))
    if cxt_len > 1 and plot:
        print(Counter(answer_indices).most_common())
        label_probs = F.one_hot(torch.LongTensor(answer_indices))
        _ = plt.figure(figsize=(10, 0.7))
        _ = sns.heatmap(label_probs.T, cbar=False); plt.show()
    all_vocabs, all_examples = (vocabs, examples) if counter_paired else ([vocabs], [examples])
    ret = []
    for vocabs, examples in zip(all_vocabs, all_examples):
        examples, text, bos_tokens = make_input_str(task, vocabs, examples,
            rev_item2str=rev_item2str, abstract=abstract, tokenizer=tokenizer)
        if verbose: print(text)
        ret.append((examples, text, bos_tokens))
    return ret if counter_paired else ret[0]

def replace_keys(text, replacements, replacements_do_neg, do_negate):   
    if do_negate ==False:
        pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in replacements.keys()) + r')\b')  
        result = pattern.sub(lambda match: replacements[match.group(0)], text)  
    else:
        pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in replacements_do_neg.keys()) + r')\b')  
        result = pattern.sub(lambda match: replacements_do_neg[match.group(0)], text)  
    return result  

replacements = {  
    "ans": "A",  
    "query": "Q",
    "bos": "END",  
    "ans0": "V",
    "tgt": "K",
    "ntgts": "K'",
    "nans0s": "V'",
    "rel": "R",
} 
replacements_do_neg = {  
    "ans": "A'",  
    "query": "Q",
    "bos": "END",  
    "ans0": "V'",
    "tgt": "K'",
    "ntgts": "K",
    "nans0s": "V",
    "rel": "R", 
} 

def main(nrows, k_shot, cxt_len, batch_size, cache_dir, model_name, proxies, device, only_load_tokenizer):
    tokenizer = load_model(cache_dir, model_name, proxies, device, only_load_tokenizer)
    
    rel1_kwargs = {'x_f': None}
    for task,rel0_i, rel1_i, do_swap_qa, do_negate, do_rm_query, rev_item2str, do_g2c in product(
        tasks[0:10],[0,1],[0,1],[False,True],[False,True],[False],[False,True],[False,'counter_paired']):

        set_seed(42)
        args = dict(cxt_len=cxt_len, rev_item2str=rev_item2str, abstract=False)
        trans_args = dict(rel0_i=rel0_i, rel1_i=rel1_i, rel1_kwargs=rel1_kwargs, do_swap_qa=do_swap_qa, do_negate=do_negate,
                        do_rm_query=do_rm_query, do_g2c=do_g2c)
        task = transform_and_validate_task(task, **trans_args, **args)
        if task is None: continue
        res_key = f'{task2str(task)}[{args2str(args)}]'  # {composed_heads2str(model)}
        print(f'\n== {res_key} == {args2str(trans_args)}')
        
        tuples = [generate(task, nrows=nrows, counter_paired=do_g2c == 'counter_paired', tokenizer=tokenizer,
                plot=False, verbose=False, **args) for _ in range(batch_size)]
        if do_g2c == 'counter_paired': tuples = join_lists(tuples)
        all_examples, texts, all_bos_tokens = zip(*tuples)
        _, _, ranges, *args = map(list, zip(*[make_data_tuple(
                text, examples, tokenizer, k_shot=k_shot, bos_tokens=bos_tokens, eos_tokens=None)
                for i, (examples, text, bos_tokens) in enumerate(zip(all_examples, texts, all_bos_tokens))]))
        for Text,Range in zip(texts, ranges):
            Ranges = [i.__dict__ for i in Range]
            for i in range(len(Ranges)):
                for key, value in Ranges[i].items():
                    if isinstance(value, tuple):
                        Ranges[i][key] = tuple(item.tolist() if isinstance(item, np.ndarray) else item for item in value)
            Trans_Ranges = []
            for i in range(len(Ranges)):
                Trans_Ranges.append({})
                for key, value in Ranges[i].items():
                    if key in ["dtgt","dans0","sep","cls","ans0s"]: continue
                    key_trans = replace_keys(key, replacements, replacements_do_neg, do_negate)
                    Trans_Ranges[i][key_trans]=Ranges[i][key]
            results = {'task_id': res_key, 'texts': Text, 'ranges': Trans_Ranges}
            
            if os.path.exists("./GAR_data.jsonl"):
                with open("./GAR_data.jsonl", 'a') as file:
                    json.dump(results, file)
                    file.write('\n') 
            else:
                with open("./GAR_data.jsonl", 'w') as file:
                    json.dump(results, file)
                    file.write('\n') 

 
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--nrows", type=int, required=False, default=2)
    parser.add_argument("--k_shot", type=int, required=False, default=1)
    parser.add_argument("--cxt_len", type=int, required=False, default=3)
    parser.add_argument("--batch_size", type=int, required=False, default=8)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--proxies", type=str, required=False, default=None)
    parser.add_argument("--device", type=str, required=False, default=0)
    parser.add_argument("--only_load_tokenizer", type=bool, required=False, default=True)
    
    args = parser.parse_args()
    main(nrows=args.nrows, k_shot=args.k_shot, cxt_len=args.cxt_len, batch_size=args.batch_size, cache_dir=args.cache_dir,
        model_name=args.model_name, proxies=args.proxies, device=args.device, only_load_tokenizer=args.only_load_tokenizer)