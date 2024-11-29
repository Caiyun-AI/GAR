import types
import random
import itertools
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from data import *
from dataclasses import dataclass 
from dataclasses import fields
from copy import deepcopy
from random import choice, shuffle, sample
from itertools import chain
try: from collections import Iterable
except Exception: from collections.abc import Iterable  # for Python >= 3.10
from functools import partial
from transformers import LlamaTokenizer
from collections import defaultdict, OrderedDict, Counter
NEW_LINE = '\n'

class InvalidTransException(Exception): pass

class Relation(object):
    def __init__(self, name, _dict):
        self.name = name
        self.verbalizer = None  # for child and sibling relations
        self._dict = _dict
        self._inv_dict = None
        self.inv_rel = None
        self.neg_rel = None
        self.x_f = None
        self.y_f = None
        self.skip_inv_f = False

    def f(self, x): return self._dict.get(x, [])
    def inv_f(self, x): return self._inv_dict.get(x, [])
    def dom(self, xs=None):
        r = list(self._dict.keys())
        return r
    def codom(self, ys=None):
        elems = join_lists(self._dict.values()) if self.name != 'sibling' else list(self._dict.keys())
        if self.name in ['parent', 'similar', 'opposite']: elems = list(dict.fromkeys(elems))
        else: assert len(elems) == len(set(elems)), f'{self.name} {len(elems)} != {len(set(elems))}'
        return elems

    def b(self, x0, x1): return x1 in self._dict.get(x0, [])

    def __str__(self):
        s = self.name if not self.skip_inv_f else 'equal'
        def attr2str(name):
            value = getattr(self, name)
            return f'{name}={value.__name__}' if value != True else name
        attr_str = ','.join(attr2str(name) for name in ['x_f', 'y_f']
                                        if getattr(self, name) not in [None, False])
        if attr_str != '': s += f'[{attr_str}]'
        return s

class NegativeRelation(Relation):
    def __init__(self, rel, set_obj):
        self.rel = self.neg_rel = rel
        rel.neg_rel = self
        self.name = 'neg_' + rel.name if not rel.name.startswith('neg_') else rel.name[4:]
        self.verbalizer = rel.verbalizer if rel.name in ['child', 'sibling'] else None
        for name in ['x_f', 'y_f', 'skip_inv_f']: setattr(self, name, getattr(self.rel, name))
        if self.name == 'neg_equal' and hasattr(set_obj, 'sibling'):  # set_obj is a TreeSet or SymSet
            self.sibling = set_obj.sibling  # used in distractive_sample

    def f(self, x): return list_diff(self.rel.codom(), self.rel.f(x))
    def inv_f(self, x): return list_diff(self.rel.dom(), self.rel.inv_f(x))
    def dom(self, xs=None): return self.rel.dom()
    def codom(self, ys=None): return self.rel.codom()
    def b(self, x0, x1): return not self.rel.b(x0, x1)

class Set(object):
    def __init__(self, data, rel_names):
        self.data = data
        self.rel_names = rel_names
        for rel_name in self.rel_names:
            setattr(self, rel_name, Relation(name=rel_name, _dict=defaultdict(list)))

    def use(self, rel_names, x_f=None, y_f=None, skip_inv_f=False):
        if isinstance(rel_names, str): rel_names = [rel_names]
        self.relations = [getattr(self, rel_name) if not rel_name.startswith('neg_') else 
            NegativeRelation(getattr(self, rel_name.replace('neg_', '')), self) for rel_name in rel_names]
        for rel in self.relations[:1]:  # TODO: check compatibility with NegativeRelation
            rel.x_f, rel.y_f, rel.skip_inv_f = x_f, y_f, skip_inv_f
        return self

    def negate_used(self):
        self.relations = [NegativeRelation(rel, self) for rel in self.relations]
        return self

    def __str__(self):
        return f"{self.data.__name__}.{self.__class__.__name__}.{'|'.join(str(rel) for rel in self.relations)}"

class SymSet(Set):
    def __init__(self, data):
        super().__init__(data, ['similar', 'opposite', 'sibling', 'equal'])
        data, verbalizers = data()
        for pair in data:
            for similars, opposites in [(pair[0], pair[1]), (pair[1], pair[0])]:
                for e in similars:
                    self.equal._dict[e] = [e]
                    if len(similars) > 1: self.similar._dict[e] = list_diff(similars, [e])
                    self.opposite._dict[e] = opposites[:]
                    self.sibling._dict[e] = list_diff(similars, [e]) + opposites  # used by neg_equal
        self.opposite._inv_dict, self.similar._inv_dict = self.opposite._dict, self.similar._dict
        self.equal._inv_dict = self.equal._dict
        self.sibling._inv_dict = self.sibling._dict
        self.similar.inv_rel, self.opposite.inv_rel = self.similar, self.opposite
        self.equal.inv_rel = self.equal
        self.sibling.inv_rel = self.sibling

class TreeSet(Set):
    def __init__(self, data):
        super().__init__(data, ['child', 'parent', 'sibling', 'equal'])
        data, verbalizers = data()
        for k, v in verbalizers.items(): getattr(self, k).verbalizer = v  # child, sibling
        for parent, children in data.items():
            self.child._dict[parent] = children
            # self.equal._dict[parent] = [parent]
            for child in children:
                self.parent._dict[child] = [parent]
                self.equal._dict[child] = [child]
                self.sibling._dict[child] = list_diff(children, [child])
        self.child._inv_dict, self.parent._inv_dict = self.parent._dict, self.child._dict
        self.sibling._inv_dict = self.sibling._dict
        self.equal._inv_dict = self.equal._dict
        self.child.inv_rel, self.parent.inv_rel = self.parent, self.child
        self.sibling.inv_rel = self.sibling
        self.equal.inv_rel = self.equal

@dataclass
class Ranges:
    bos: tuple = None
    ans: tuple = None
    cls: tuple = None
    ans0: tuple = None
    query: tuple = None
    tgt: tuple = None
    dans0: tuple = None
    dtgt: tuple = None
    rel: tuple = None
    sep: tuple = None
    ans0s: list = None
    ntgts: list = None
    nans0s: list = None
    example: tuple = None

@dataclass
class Result:
    task: tuple = None
    trans_args: dict = None
    gen_args: dict = None
    all_examples: list = None
    texts: list = None
    all_bos_tokens: list = None
    data_tuples: list = None
    mean_loss: float = None
    mean_acc: float = None

def join_lists(x, dedup=False):
    l = list(chain.from_iterable(x))
    if dedup: l = list(OrderedDict.fromkeys(l)) # list(set(l)) # to keep order
    return l

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def list_diff(l1, l2):  # will preserve order of elements in l1 compared to list(set(l1) - set(l2))
    l2 = set(l2)
    return [x for x in l1 if x not in l2]

def my_isinstance(obj, type_):  # to cope with autoreload
    # return isinstance(obj, type_)  # fail on autorelaod
    return obj.__class__.__name__ == type_.__name__ if not isinstance(type_, tuple) \
        else any(obj.__class__.__name__ == t.__name__ for t in type_)

def split_examples(examples):
    assert len(examples) % 2 == 0
    nrows = len(examples) // 2
    pairs = [[examples[i * 2], examples[i * 2 + 1]] for i in range(nrows)]
    k_shot = nrows - 1
    k_shot_examples = [choice(p) for p in pairs[:k_shot]]
    return k_shot_examples + [pairs[-1][0]], k_shot_examples + [pairs[-1][1]]

def make_examples(task, nrows=4, vocab_for_each_row=False, counter_paired=False, **kwargs):
    vocab_fn, example_gen_fn = task[:2]
    vocabs, examples = [], []
    qa_set = set() # for dedup
    if any(v.__class__.__name__ == 'PoSet' for v in vocab_fn()):
        vocab_for_each_row = True
    if not vocab_for_each_row: vocab = vocab_fn()
    for i in range(nrows * (1 + int(not(counter_paired)))):
        if vocab_for_each_row: vocab = vocab_fn()
        example_or_pair = example_gen_fn(vocab, **kwargs)
        if counter_paired:  # example_or_pair is a pair
            examples += example_or_pair
            vocabs.append(vocab)
        else:  # example_or_pair is an example
            cxt, query, candidates, ans_chain, *a = example_or_pair
            if isinstance(query, list): query = tuple(query)
            if (tuple(cxt), query, ans_chain) not in qa_set:
                qa_set.add((tuple(cxt), query, ans_chain))
                vocabs.append(vocab)
                examples.append([cxt, query, candidates, ans_chain, *a])
            if len(examples) == nrows: break
    if counter_paired:
        vocabs, examples = [vocabs] * 2, split_examples(examples)
    return vocabs, examples

def candidates2dict(candidates, is_cls=False, names=['query', 'tgt', 'ans0', 'ans']):
    if isinstance(candidates, OrderedDict): return candidates
    assert len(candidates) in [4, 5, 6, 7], f'len({candidates}) == {len(candidates)}'
    if len(candidates) <= 5:  # rlr
        names = ['query', 'tgt', 'ans0', 'ans'] 
        _rel_hops = [('query', 'tgt'), ('ans', 'ans0')]
    else: # rlrlr
        names = ['query', 'tgt', 'ans12', 'parent', 'ans0', 'ans']
        _rel_hops = [('query', 'tgt'), ('parent', 'ans12'), ('ans', 'ans0')]
    if len(candidates) in [5, 7]:  # g2c: 4 + 1 （rlr）or 6 + 1 (rlrlr)
        assert len(candidates[-1]) == 2, str(candidates[-1])
        is_cls = True
        names = names + ['cls']
    return OrderedDict(zip(['_rel_hops'] + names, [_rel_hops] + candidates))

def split_answer(text, bos_token):
    i = text.rindex(bos_token) + len(bos_token)
    return text[:i], text[i:].strip()

def get_answer_index(example):
    cxt, query, cands, (*_, ans), *cls = example
    # if len(cxt) <= 1: return 0  # for cxt_len==1 + ~has_local_hop + g2c
    cands = candidates2dict(cands)
    return list(cands.values())[-1].index(ans if len(cls) == 0 else cls[0])

def get_rel_candidates(candidates, use_codom=False):
    def try_join(l): return join_lists(l) if isinstance(l[0], tuple) else l # for len(vocabs[1].relations) > 1 in rlr_gen
    return [try_join(candidates[rh[int(use_codom)]]) for rh in candidates['_rel_hops']]

def fn2str(fn, excluded_keys=[]):
    if isinstance(fn, types.FunctionType): return fn.__name__
    assert isinstance(fn, partial), str(fn)
    def convert_value(k, v):
        if k in excluded_keys: return '...'
        if isinstance(v, torch.Tensor): return v.size()
        if isinstance(v, types.FunctionType): return v.__name__
        return v
    return fn.func.__name__ + '[' + ','.join(f'{k}={convert_value(k, v)}' for k, v in fn.keywords.items()) + ']'

def task2str(task):
    vocab_fn, gen_fn, *_ = task
    return f"{fn2str(gen_fn)}[{','.join(str(v) for v in vocab_fn())}]"

synonym_dict = {
    # 'has': ['owns', 'possesses'], 'have': ['own', 'possess'],
    # 'wants to go to': ['wants to visit', 'longs for', 'yearns for'], 'want to go to': ['want to visit', 'long for', 'yearn for'], 
    'arrived': ['appeared', 'showed up'], 'arrive': ['appear', 'show up'],
}

synonym_dict = {k: [k] + v for k, v in synonym_dict.items()}

def sampled_synonym_dict(): return {k: choice(v) for k, v in synonym_dict.items()}

def multi_replace(s, pairs):
    for old, new in pairs.items():
        if old in s: s = re.sub(r"\b%s\b" % old, new, s)
    return s

def a_(noun):  # prepend indefinite article a/an if possible
    if noun[0].isupper() or noun in adjs:
        return noun
    
    d = {'apple':  'an apple',  'chip': 'chips',  'coffee': 'coffee',  'biscuit': 'biscuits', 'dog': 'a dog', 'tea': 'tea'}
    if noun in d: return  d[noun]

def strip_a(text):
    if text.startswith('a ') or text.startswith('an '):
        text = re.sub(r"^a ", "", text); text = re.sub(r"^an ", "", text)
    return text

def lget(l, i, default=None): return l[i] if l is not None and len(l) > i else default

def rsplit_bos(s):
    if s.endswith("'s"): return "'s"
    if s.endswith("?"): return "?"
    return ' ' + s.split()[-1]

def post_compose(fn, fn2):
    def new_fn(*args, **kwargs):
        return fn2(fn(*args, **kwargs))
    return new_fn

def try_wh_question2statement(s, vocab):  # convert wh-questions brought by swap_qa to statement
    if not hasattr(vocab.data, 'wh'): return s
    wh, sub_wh = vocab.data.wh + ' ', vocab.data.sub_wh + ' '
    if wh in s and sub_wh not in s:
        assert '?' in s, s
        s = s.replace(wh, sub_wh)  # who has apple? -> the one who has apple?
        for old, new in [('who is', "who's"), ('what is', "who's")]: s = s.replace(old, new)
        assert ' is ' not in s, s  # avoid two "is" in sent, which would cause bug in negate_sent
        s = s.replace('?', ' is')  # the one who has apple? -> the one who has apple is
        return s
    return s

def _rel_cands2str(rel_candidates, vocabs, i=1, verb='include'):
    rel_cands, vocab = rel_candidates[i], vocabs[i]
    rel_cands = sample(rel_cands, len(rel_cands))
    def join_fn(cands): return ', '.join(cands[:-1]) + ' and ' + cands[-1]
    return f"{capitalize(vocab.data.name)} {verb} {join_fn(rel_cands)}." \
        if hasattr(vocab.data, 'name') else ''

def capitalize(s):  # different from str.capitalize() in more than one way!
    if s.startswith(' '): return ' ' + capitalize(s[1:])
    return s[0].upper() + s[1:] if s else ''

def the_(noun, uppercase=False):
    if noun.lower() in ['who', 'which']:   # in swap_qa
        return capitalize(noun) if uppercase else noun
    if noun[0].isupper(): return noun  # proper noun
    the = 'The' if uppercase else 'the'
    return the + ' ' + strip_a(noun)


def _item2str(item, vocab=None): #, reverse=False):
    return [f'{item[0]} {item[1]}', f'{item[1]} {item[0]}'] if isinstance(item, tuple) else f'{item}'

def _cxt2str(cxt, vocab=None, prefix='< ', suffix=' >.', sep=' ', item2str=_item2str, rev_item2str=False):
    def try_wrap(s): return [s] if type(s) == str else s
    return prefix + sep.join([try_wrap(item2str(item, vocab))[int(rev_item2str)] for item in cxt]) + suffix

def args2str(args):
    strs = []
    for k, v in args.items():
        if type(v) == dict: s = f'{k}=({args2str(v)})' if args2str(v) != '' else ''
        elif v is None: s = ''
        elif type(v) == bool: s = k if v else ''
        elif type(v) == int: s = f'{k}={v}' if v != 0 else ''
        elif type(v) == types.FunctionType: s = f'{k}={v.__name__}'
        else: s = f'{k}={v}'
        strs.append(s)
    return ','.join(s for s in strs if s != '')

def locate(whole_string, tokens, substring, return_last=False, return_all=False):
    if substring is None: return None
    substring = substring.lower() 
    substring = strip_a(substring)
    assert substring in whole_string, f'{tokens}\n{substring} not in {whole_string}'
    if substring.strip() in ['->', '?', ':']:
        char_locations = [whole_string.index(substring), whole_string.rindex(substring)]
    else:
        pattern = r"(?<!-)\b%s(?:s|es)?" if not substring.startswith(" ") else r"%s(?:s|es)?"  
        if substring[-1] not in ['.']: pattern = pattern + r"\b"
        try: matches = list(re.finditer(pattern % substring, whole_string))
        except Exception: print(f'sub = {substring}, whole = {whole_string}'); raise
        assert len(matches) > 0, f'{tokens}\n{substring} not match {whole_string}'
        char_locations = [m.span()[0] for m in matches]
    if not return_all: char_locations = [char_locations[-int(return_last)]]
    ranges = []
    for char_loc in char_locations:
        loc = 0; tok_start, tok_end = None, None
        for i, t in enumerate(tokens):
            loc += len(t)
            _t = t[1:] if t.startswith(' ') else t
            forms = [substring, substring + 's', substring + 'es']
            if tok_start is None and loc > char_loc:
                assert any(s.find(_t) in [0, 1] for s in forms), \
                    f'{whole_string}\n{tokens}\n{substring} not startswith {_t} at {i}. loc = {loc}, char_loc = {char_loc}'
                tok_start = i
            if tok_end is None and loc >= char_loc + len(substring):
                assert any(s.endswith(_t) for s in forms), \
                    f'{whole_string}\n{tokens}\n{substring} not endswith {_t} at {i}'
                tok_end = i + 1
                break
        assert tok_start is not None and tok_end is not None, f'{tok_start}, {tok_end}'
        if not return_all: return (tok_start, tok_end)
        ranges.append((tok_start, tok_end))
    return ranges

def example2ranges(example, tokens, bos_token, case_sensitive=False, trimmed=False):
    cxt, query, candidates, (tgt, *others, ans0, ans), *cls = example
    cls = cls[0] if len(cls) > 0 else None
    if trimmed:
        ranges = Ranges(bos = locate(tokens, bos_token, return_last=True))
        ranges.bos = (ranges.bos[1] - 1, ranges.bos[1])
        return ranges
    if not case_sensitive: tokens = [t.lower() for t in tokens]
    whole_string = "".join(t for t in tokens)
    rel_word = None # 'capital'  # TODO: systematic treatment of rel_word, must be lowercase
    if ' capital ' in whole_string: rel_word = 'capital'
    elif ' not ' in whole_string: rel_word = 'not'
    
    ranges = Ranges(
        bos = locate(whole_string, tokens, bos_token, return_last=True),
        ans = locate(whole_string, tokens, ans, return_last=True),
        cls = locate(whole_string, tokens, cls, return_last=True),
        ans0 = locate(whole_string, tokens, ans0),
        query = locate(whole_string, tokens, query, return_last=True),
        tgt = locate(whole_string, tokens, tgt),
        rel = locate(whole_string, tokens, rel_word, return_last=True) if rel_word is not None and rel_word in whole_string else None,
        example = (0, len(tokens))
    )
    if cls == None:
        ranges.ans = tuple([ranges.bos[1],ranges.ans[1]])
    if len(others) > 0 and cls is not None:
        assert len(others) in [2, 4], str(others) # len(others) == 4 for rlrlr
        dtgt, dans0 = others[:2]
        ranges.dtgt = locate(whole_string, tokens, dtgt)
        ranges.dans0 = locate(whole_string, tokens, dans0)
    ranges.bos = (ranges.bos[1] - 1, ranges.bos[1])
    if len(cxt) == 0: return ranges  
    if candidates is not None:
        candidates = candidates2dict(candidates, cls is not None)
        ans0s = candidates['ans0']
        max_i = ranges.query[0] if ranges.query is not None else ranges.ans[0]
        ranges.ans0s = tuple(map(np.array, zip(*filter(lambda x: x[0] < max_i, join_lists(
            [locate(whole_string, tokens, a0, return_all=True) for a0 in ans0s], dedup=True)))))
        ranges.nans0s = tuple(map(np.array, zip(*filter(lambda x: x[0] < max_i, join_lists(
            [locate(whole_string, tokens, a0, return_all=True) for a0 in ans0s if a0 != ans0], dedup=True)))))
        ranges.ntgts = tuple(map(np.array, zip(*filter(lambda x: x[0] < max_i, join_lists(    
            [locate(whole_string, tokens, t, return_all=True) for t in candidates['tgt'] if t != tgt], dedup=True)))))
    if ranges.tgt is not None and '.' in tokens[ranges.tgt[1]:]:  # TODO: debug
        sep_i = tokens.index('.', ranges.tgt[1])
        ranges.sep = (sep_i, sep_i + 1)
    return ranges

def distractive_sample(cxt_len, rel, n_answers=1):
    query = choice(rel.dom())
    siblings = rel.sibling.f(query) if rel.name == 'neg_equal' and hasattr(rel, 'sibling') else []
    answers = sample(list_diff(rel.f(query), siblings), n_answers)
    distractors = list_diff(rel.codom(), rel.f(query) + ([query] if rel.name == 'sibling' else []))
    k = cxt_len - n_answers
    # neg_xxx or parent rel may have only one distractor
    assert len(distractors) >= k or len(distractors) == 1, \
        f'{rel.name}, query = {query}, f(query) = {rel.f(query)}, distractors = {distractors}'
    distractors = sample(distractors, k) if len(distractors) >= k else distractors * k
    # TODO: rel.inv_f(x)[0] -> choice(rel.inv_f(x)). check not equivalent for sibling
    distractors0 = [choice(rel.inv_f(x)) for x in distractors]
    candidates = [[query] * n_answers + distractors0, answers + distractors]
    assert len(candidates[0]) == len(candidates[0]), f'{len(candidates[0])} != {len(candidates[0])}'
    if rel.skip_inv_f and rel.x_f is None: rel.x_f = lambda x: x
    if rel.x_f: candidates[0] = [rel.x_f(c) for c in candidates[int(rel.skip_inv_f)]]
    if rel.y_f: candidates[1] = [rel.y_f(c) for c in candidates[1]]
    return candidates
    # return tuple([swap(l, ans_i) for l in candidates])

def rlr_gen(vocabs, cxt_len=3, cxt_sample_fn=None, query=None, use_numpy=False, dict_candidates=False):
    rels = [v.relations[0] for v in vocabs]
    fixed_query = query is not None
    has_local_hop = vocabs[0].data != vocabs[1].data
    position_relevant = getattr(cxt_sample_fn, '__name__', None) == 'enumerate_sample'
    
    sample_fn = distractive_sample if not fixed_query else cxt_sample_fn
    candidates = sample_fn(cxt_len, rels[0]) # hop0: query_cands, tgt_cands
    cand_keys = ['query', 'tgt']
    candidates += distractive_sample(cxt_len, rels[1])[::-1] if has_local_hop \
        else (candidates[-1].copy(), [choice(rels[1].inv_f(x)) for x in candidates[-1]])  # hop2: ans0_cands, ans_cands
    cand_keys += ['ans0', 'ans']
    if len(vocabs[1].relations) > 1:
        assert not use_numpy
        for r in vocabs[1].relations[1:]:  # typically opposite
            def cat(a, b): return a + (b,) if isinstance(a, tuple) else (a, b)
            candidates[-1] = [cat(c, choice(r.inv_f(x))) for x, c in
                              zip(candidates[-2], candidates[-1])]
    i = 0 if query is None else candidates[0].index(query)
    if use_numpy: candidates = np.array(candidates)  # # -> 4 * cxt_len array
    else: tuples = list(zip(*candidates))  # row -> col

    query, *ans_chain = tuples[i] # candidates[:, i]
    ans_chain = tuple(ans_chain)
    if not position_relevant:
        if use_numpy: candidates = candidates[:, np.random.permutation(cxt_len)]
        else: shuffle(tuples)
    if use_numpy:
        cxt = list(map(tuple, candidates[int(not fixed_query):3].T))  # hop1: tgt_cands, ans0_cands
        candidates = candidates.tolist()
    else:
        cxt = [t[int(not fixed_query):3] for t in tuples]  # hop1: tgt_cands, ans0_cands
        candidates = [list(c) for c in zip(*tuples)]  # col -> row

    if fixed_query: cxt, query = [x[1:] for x in cxt], None
    if not has_local_hop: cxt = [x[0] for x in cxt]
    if dict_candidates:
        _rel_hops = [('query', 'tgt'), ('ans', 'ans0')]  # hop0/2
        candidates = OrderedDict(zip(['_rel_hops'] + cand_keys,
                                     [_rel_hops] + candidates))
    return cxt, query, candidates, ans_chain


def move_ranges(r, offset): 
    for field in fields(r):
        name = field.name; pair = getattr(r, name)
        if pair is not None: setattr(r, name, tuple([i + offset for i in pair]))
    return r

def _str(l, vocab=None, sep=' '):
    if l is None: return ''
    if isinstance(l, str) or not isinstance(l, Iterable): l = [l]
    # l = [e for e in l if not my_isinstance(e, Sequence)] #type(e).__name__ != 'Sequence']
    if isinstance(l, (dict, OrderedDict)): l = [f'{k}: {v}' for k, v in l.items()]
    return sep.join(str(i) for i in l)

def negate_sent(s, vocabs):  # TODO: need better way of negating a sentence
    if s.startswith('So '): s = s[3:]
    s = try_wh_question2statement(s, vocabs[1])
    s00 = s0 = s
    n_replaced = 0
    for old, new in [
        # (" likes", " does not like"), (" owns", " does not own"), (" possesses", " does not possess"),
        (r"\bis\b", "is not"),(r"\bhas\b", "does not have"),(r"\bowns\b", "does not own"), 
        (" wants ", " does not want "), #(" wanna ", " does not want to "),
        (" arrived", " did not arrive")]:
        if r"\b" in old: s = re.sub(old, new, s0)
        else: s = s0.replace(old, new)
        if s != s0:
            n_replaced += 1
            # skip 'has' if 'is' is present. e.g. 'What John has is' -> 'What John has is not'
            if old == r"\bis\b": break
        s0 = s
    assert n_replaced == 1, f'n_replaced = {n_replaced}: {s00} -> {s}'

    singular_subs, plural_subs = ['boy ', 'girl '], ['boys ', 'girls ']
    sep = r"\bnot\b" #if " not " in s else r"\bis\b"
    sep_i = list(re.finditer(sep, s))[0].span()[0]
    if any(sub in s[:sep_i] for sub in singular_subs):
        for old_sub, new_sub in zip(singular_subs, plural_subs):
            s = s.replace(old_sub, new_sub)
        s = s.replace(" does not", " do not")
    return s

def verbalize_relation(vocab):
    data_name, rel_name = vocab.data.__name__, vocab.relations[0].name
    _rel_name = rel_name.split('neg_')[-1]
    verbalizer = vocab.relations[0].verbalizer
    if _rel_name in ['child', 'sibling'] and verbalizer:
        rel_str = ' ' + verbalizer
    else: rel_str = ''
    return rel_str

def locate_bos(tokens, bos_token):
    bos_indices = []
    found_in_example = False
    for i, token in reversed(list(enumerate(tokens))):
        if token == bos_token and not found_in_example:
            bos_indices.insert(0, i)
            found_in_example = True
        elif token == '\n':
            found_in_example = False
    return bos_indices

def locate_answers(input_ids, tokenizer, bos_indices=None, bos_tokens=None, eos_tokens=None,
        space_token='Ġ', nrows=None):
    if bos_tokens is not None:
        assert all(t == bos_tokens[0] for t in bos_tokens)
        bos_token = bos_tokens[-1]
    if isinstance(tokenizer, types.FunctionType):  # api models
        def pad(l): return [None] * (nrows - 1) + l
        if tokenizer.is_chat:
            input_ids, answer = split_answer(input_ids, bos_token)
            _, logits = tokenizer(input_ids)
            return pad([-1]), pad([1]), pad([[answer]]), logits # sue labels to pass logits
        tokens, logits = tokenizer(input_ids)
        assert bos_tokens is not None
        if bos_token not in tokens: # e.g. bos_token=owns (got by rsplit_bos), tokens=[own, s]. Use api tokenizer to get the correct bos.
            input_without_ans, answer = split_answer(input_ids, bos_token)
            bos_token = tokenizer(input_without_ans)[0][-1]
        assert bos_token in tokens, f"bos_token = {bos_token}, tokens = {tokens}"
        bos_indices = locate_bos(tokens, bos_token)
        if len(bos_indices) != nrows:
            assert len(bos_indices) == nrows + 1, f'{len(bos_indices)} > {nrows}'
            bos_indices = bos_indices[-nrows:]

        eos_indices = [bos_i + 2 for bos_i in bos_indices]
        answers = [tokens[bos_i + 1: eos_i] for bos_i, eos_i in zip(bos_indices, eos_indices)]
        # labels = torch.ones((1, len(tokens)), dtype=torch.long) * (-100)
        # return pad(bos_indices), pad(eos_indices), pad(answers), labels
        return bos_indices, eos_indices, answers, logits  # use labels to pass logits
    assert input_ids.size(0) == 1  # bsz == 1
    if bos_indices is None:
        bos_id = tokenizer.convert_tokens_to_ids(bos_token.replace(' ', space_token))
        bos_indices = (input_ids[0] == bos_id).nonzero().squeeze(1).tolist()#[1:]
    if nrows is not None:
        assert nrows == len(bos_indices)
    else:
        nrows = len(bos_indices)
    if eos_tokens is not None:
        assert all(t == eos_tokens[0] for t in eos_tokens)
        eos_id = tokenizer.convert_tokens_to_ids(eos_tokens[0])
        eos_indices = (input_ids[0] == eos_id).nonzero()[-nrows:].squeeze(1).tolist()
    else:
        # eos_indices = bos_indices[1:] + [input_ids.size(1)]
        eos_indices = [bos_i + 2 for bos_i in bos_indices]
    labels = torch.ones_like(input_ids) * (-100)
    answers = []
    for bos_i, eos_i in zip(bos_indices, eos_indices):
        ans_ids = input_ids[0, bos_i + 1: eos_i]
        labels[0, bos_i: eos_i - 1] = ans_ids
        answers.append(ans_ids.numpy())
    return bos_indices, eos_indices, answers, labels

def locate_ranges(examples, example_strs, tokenizer, input_ids, bos_token, instruction=None):
    assert len(examples) == len(example_strs)
    ranges = []
    use_llama_tokenizer = my_isinstance(tokenizer, LlamaTokenizer)
    is_yi_tokenizer = 'Yi-34B' in tokenizer.name_or_path
    newline_token = '<0x0A>' if use_llama_tokenizer and not is_yi_tokenizer else '\n' 
    if use_llama_tokenizer: 
        # tokenizer.decode will strip leading '__'
        all_tokens_llama = [tokenizer.convert_ids_to_tokens(id).replace('▁', ' ') for id in input_ids]
        all_tokens = []
        if not is_yi_tokenizer:  
            assert all_tokens_llama[0] == tokenizer.bos_token, str(all_tokens_llama)
            all_tokens = [tokenizer.bos_token]
            assert all_tokens_llama[1].startswith(' '), all_tokens_llama[1]
            all_tokens_llama[1] = all_tokens_llama[1][1:]
            all_tokens_llama = all_tokens_llama[1:]  # treat leading bos as prefix_token and remove it from all_tokens_llama
        # split all_tokens_llama using newline_token as delimiter
        # https://stackoverflow.com/questions/15357830/splitting-a-list-based-on-a-delimiter-word
        sep_tokens = [list(y) for x, y in itertools.groupby(all_tokens_llama, lambda z: z == newline_token) if not x]
        if instruction:
            inst_tokens, *sep_tokens = sep_tokens
            assert ''.join(inst_tokens) == instruction, f"{inst_tokens} -> {''.join(inst_tokens)} != {instruction}"
            all_tokens += inst_tokens + [newline_token]
        assert len(sep_tokens) == len(example_strs), f'{len(sep_tokens)} != {len(example_strs)}'
    else:
        all_tokens = [newline_token] if tokenizer.decode(input_ids[0]) == newline_token else []
        # TODO: deal with instruction
    for i, (e, e_str) in enumerate(zip(examples, example_strs)):
        tokens = sep_tokens[i] if use_llama_tokenizer else \
            [tokenizer.decode(id) for id in tokenizer.encode(e_str)] 
        assert ''.join(tokens) == e_str, f"{tokens} -> {''.join(tokens)} != {e_str}"
        r = example2ranges(e, tokens, bos_token[i] if isinstance(bos_token, (tuple, list)) else bos_token)
        ranges.append(move_ranges(r, len(all_tokens)))
        all_tokens += tokens + [newline_token]
    return ranges

def make_input_str(task, vocabs, examples, rev_item2str=False, abstract=False, options_position=None, tokenizer=None):
    # Randomized transformations here are per input basis, i.e. each example in an input are the same,
    # while each input in a task's batch may be different. It is finer-grained than transform_task which are per task basis.
    # Hierarchy: task >= batch > input > example
    cxt, *_ = examples[0]
    cxt_len = len(cxt)
    instruction, cxt2str, query2str, bos_token, ans2str = \
        [lget(task, i, '' if i in [2, 5] else _str) for i in range(2, 7)]
    if isinstance(instruction, tuple): instruction, rel_cands2str = instruction
    elif vocabs[0][1].relations[0].name in ['child', 'similar', 'opposite']: rel_cands2str = _rel_cands2str
    else: rel_cands2str = None
    if isinstance(cxt2str, types.FunctionType) and cxt2str.__name__ == 'empty_cxt2str':
        examples = [(cxt, query, None, (None, None, ans), *cls)
            for cxt, query, candidates, (tgt, ans0, ans), *cls in examples]
    query2str = post_compose(query2str, partial(multi_replace, pairs=sampled_synonym_dict()))
    def example2str(vocab, example):
        cxt, query, candidates, (*_, ans), *cls = example
        cxt_str = cxt2str(cxt, vocab=vocab, rev_item2str=rev_item2str)
        query_str, ans_str = capitalize(query2str(query, vocab)), ans2str(ans)
        strs = [cxt_str, query_str]
        if options_position is not None: assert False 
        s = ' '.join(s for s in strs if s != '') + bos_token + ' ' + ans_str
        _bos_token = bos_token
        if bos_token == '':
            data, rel_name = vocab[1].data, vocab[1].relations[0].name
            # TODO: data.bos is unnecessary. If bos is the, it's better to put it in data()[1]['child]
            # so that we could always get it by rsplit_bos(query_str)
            # _bos_token = rsplit_bos(query_str)
            _bos_token = data.bos[rel_name] if hasattr(data, 'bos') and rel_name in data.bos else rsplit_bos(query_str)
        # if len(cls) > 0: _bos_token = '?'; s += _bos_token + ' ' + _str(cls[0]) # g2c
        if len(cls) > 0: _bos_token = ':'; s += '? Answer' + _bos_token + ' ' + _str(cls[0]) # g2c
        return s, _bos_token
    example_strs, bos_tokens = zip(*[example2str(v, e) for v, e in zip(vocabs, examples)])
    if rel_cands2str is not None:
        rel_cands = [get_rel_candidates(candidates2dict(candidates)) for cxt, query, candidates, *_ in examples]
        joined_rel_cands = list(map(partial(join_lists, dedup=True), zip(*rel_cands)))
        rel_cands_str = rel_cands2str(joined_rel_cands, vocabs[0])  # TODO: assumes all examples share the same vocab (vocab_for_each_row == False) so we use vocab of 1st example
        instruction = ' '.join([instruction, rel_cands_str]) if instruction else rel_cands_str
    if instruction and not instruction.endswith('\n'): instruction = instruction + '\n'
    text = instruction + (NEW_LINE + ' ').join(example_strs) + '\n' \
        if isinstance(tokenizer, LlamaTokenizer) else \
        '\n' + instruction + '\n'.join(example_strs) + '\n'  # prepend '\n' to act as bos for tokenizer without bos
    return examples, text, bos_tokens

def choose_rels(task, rel_indices):
    vocab_fn, gen_fn, *a = task
    vocabs = vocab_fn()
    for hop, rel_i in enumerate(rel_indices):
        if isinstance(rel_i, int) and rel_i >= len(vocabs[hop].relations): return None

    def new_vocab_fn():
        vocabs = vocab_fn()
        for hop, rel_i in enumerate(rel_indices):
            vocabs[hop].relations = vocabs[hop].relations[rel_i: rel_i + 1] \
                if not isinstance(rel_i, Iterable) else [vocabs[hop].relations[i] for i in rel_i]
        return vocabs
    
    task = new_vocab_fn, gen_fn, *a
    return task

def decorate_rel(task, hop, kwargs):
    vocab_fn, gen_fn, *a = task

    def new_vocab_fn():
        vocabs = vocab_fn()
        rel = vocabs[hop].relations[0]
        for k, v in kwargs.items(): setattr(rel, k, v)
        return vocabs
    task = new_vocab_fn, gen_fn, *a
    return task

def get_wh_and_the(vocab):
    data_name, rel_name = vocab.data.__name__, vocab.relations[0].name
    if hasattr(vocab.data, 'wh'): wh = vocab.data.wh
    else: wh = 'who' if data_name in ['persons', 'genders_of_persons'] else 'which'
    the = '' # ' the' if data_name == 'genders_of_persons' and rel_name == 'child' or \
        # data_name == 'capabilities_of_things' and rel_name != 'child' or \
        # data_name == 'kinds_of_things' and rel_name != 'child' else ''
    return wh, the

def swap_qa(task):
    vocab_fn, gen_fn, inst, cxt2str, query2str, *a = task
    if isinstance(gen_fn, partial) and 'query' in gen_fn.keywords:
        raise InvalidTransException(f"invalid swap_qa with fixed_query = {gen_fn.keywords['query']}")
    def new_vocab_fn(): return vocab_fn()[::-1]  # would cause infinite recursion bug if use same name
    new_cxt2str = cxt2str
    if isinstance(cxt2str, partial) and 'item2str' in cxt2str.keywords:
        item2str = cxt2str.keywords['item2str']
        swapped_item2str = lambda i, v: item2str(i[::-1], v)
        new_cxt2str = deepcopy(cxt2str)
        new_cxt2str.keywords['item2str'] = swapped_item2str

    if isinstance(query2str, tuple):
        new_query2str = query2str[1]
    else:
        def new_query2str(q, v):
            wh, the = get_wh_and_the(v[1])
            return f'{query2str(wh, v)} {q}?'.replace("who's", "whose") + capitalize(the) 
    task = (new_vocab_fn, gen_fn, inst, new_cxt2str, new_query2str, *a)
    return task

def refine_query2str(task, do_swap_qa=False):
    vocab_fn, gen_fn, inst, cxt2str, query2str, *a = task
    if query2str is None: return task
    def new_query2str(q, vocabs):
        # refine_query2str.q2s is called BEFORE swap_qa.q2s, but transformed vocab_fn is called before ALL q2s.
        # So vocabs may have already been swapped by do_swap_qa and need not be swapped again here
        vocab0, vocab1 = vocabs #if not do_swap_qa else vocabs[::-1]
        return query2str((verbalize_relation(vocab0) + ' ' + q).strip(), vocabs) + verbalize_relation(vocab1)
    task = (vocab_fn, gen_fn, inst, cxt2str, new_query2str, *a)
    return task

def negate(task):
    vocab_fn, gen_fn, inst, cxt2str, query2str, *a = task

    def new_vocab_fn():
        vocabs = vocab_fn()
        return [vocabs[0].negate_used(), vocabs[1]]

    new_query2str = (lambda q, v: negate_sent(query2str(q, v), v)) \
        if query2str is not None else None

    task = (new_vocab_fn, gen_fn, inst, cxt2str, new_query2str, *a)
    return task

def remove_local_hop(task, do_swap_qa, do_rm_query, do_g2c, cxt_len):
    vocab_fn, gen_fn, inst, cxt2str, query2str, *a = task
    vocabs = vocab_fn()
    assert vocabs[0].data == vocabs[1].data
    data_name = vocabs[0].data.__name__
    rel_names = [v.relations[0].name for v in vocab_fn()]
    fixed_query = isinstance(gen_fn, partial) and 'query' in gen_fn.keywords
    
    assert not rel_names[1].startswith('neg_'), rel_names[1]
    if fixed_query:
        pass  # TODO: Is there any rule for fixed_query?
    elif cxt_len == 1 and not (rel_names[0] == 'equal' and rel_names[1] == 'equal'):
        pass
    elif do_swap_qa:
        raise InvalidTransException(f"invalid rel for rm_local_hop with swap_qa: {rel_names}")
    elif rel_names[0] == 'equal' or rel_names[1] in ['child', 'sibling'] and not rel_names[0].startswith('neg_'):
        raise InvalidTransException("invalid rel for rm_local_hop: " + str(rel_names))
    elif rel_names[1] in ['child', 'sibling'] and len(vocabs[1].child.dom()) == 2 and not do_rm_query:
        raise InvalidTransException(f"invalid rel for rm_local_hop: {rel_names}. len({data_name}.child.dom()) == 2")
    elif not do_rm_query and do_g2c:  # solvable without cxt
        raise InvalidTransException(f"invalid rel for rm_local_hop with g2c: {rel_names}")
    
    if cxt2str is None:
        cxt2str = partial(_cxt2str, prefix='There are ', suffix='.', sep=', ', item2str=lambda i, _: [a_(i), ''])
    
    if query2str is None:
        if not fixed_query:
            def query2str(q, v):
                wh, the = get_wh_and_the(v[1])
                rel_str = verbalize_relation(v[1]) + the
                neg_str = ' not' if rel_names[0].startswith('neg_') else ''
                return f"{wh} is{neg_str}{verbalize_relation(v[0])} {q}?" + capitalize(rel_str)
        else:
            query2str = lambda q, v: ""
    task = vocab_fn, gen_fn, inst, cxt2str, query2str, *a
    return task

def remove_query(task):
    vocab_fn, gen_fn, inst, cxt2str, query2str, *a = task
    vocabs = vocab_fn()
    rel_names = [v.relations[0].name for v in vocabs]
    if not rel_names[0].startswith('neg_') or rel_names[0] == 'neg_sibling':  # neg_sibling == neg_child
        raise InvalidTransException("invalid rel for rm_query" + str(rel_names))

    def new_gen_fn(*args, **kwargs):
        cxt, query, candidates, (tgt, *a, ans0, ans) = gen_fn(*args,**kwargs)
        query, candidates = None, ([None] * len(candidates[1]),) + candidates[1:]
        return cxt, query, candidates, (tgt, *a, ans0, ans)
    new_gen_fn.__name__ = f"rm_query[{fn2str(gen_fn)}]"

    def new_query2str(q, v):
        wh, the = get_wh_and_the(v[1])
        rel_str = verbalize_relation(v[1]) + the
        return f"{wh} is different?" + capitalize(rel_str)
    task = vocab_fn, new_gen_fn, inst, cxt2str, new_query2str, *a
    return task

def _g2c(g_fn, cls_labels=['Yes', 'No', 'Maybe'][:2], counter_paired=False):
    def wrapped(*args,**kwargs):
        cxt, query, candidates, (tgt, *a, ans0, ans) = g_fn(*args,**kwargs)
        vocabs = args[0]
        has_local_hop = vocabs[0].data != vocabs[1].data
        rel0, rel1 = [v.relations[0] for v in vocabs]

        def _gen(is_positive):
            _query = query  # avoid iterative modifications to query
            if len(vocabs[1].relations) > 1:
                assert isinstance(ans, tuple)
                assert len(vocabs[1].relations) == len(ans) == 2
                label, _ans = (cls_labels[0], ans[0]) if is_positive else (cls_labels[1], ans[1])
                _dtgt, _dans0 = tgt, ans0
            elif is_positive:
                label = cls_labels[0]
                _ans = ans
                _dtgt, _dans0 = tgt, ans0
            else:
                label = cls_labels[1]
                if not has_local_hop and len(cxt) == 1:  
                    _ans = choice(list_diff(rel1.dom(), [ans]))
                    # _ans0 = choice(rel1.f(_ans)) # ans0 does not occur in example_str
                    # cxt, tgt = [], None
                    _dtgt, _dans0 = None, None
                elif len(cxt) == 1:  # e.g. John has an apple. So Tom has a kind of fruit? No
                    _query = choice(list_diff(rel0.dom(), [query]))
                    _dtgt, _dans0, _ans = tgt, ans0, ans
                else:
                    _dtgt, _dans0, _ans = choice([(t, c0, c) for q, t, c0, c in zip(
                        *[candidates2dict(candidates)[k] for k in ['query', 'tgt', 'ans0', 'ans']])
                        if c != ans and (query is None or q != query)])
            _candidates = deepcopy(candidates)  # avoid in-place or iterative modifications to candidates
            if isinstance(_candidates, OrderedDict): _candidates['cls'] = cls_labels
            else: _candidates = _candidates + [cls_labels]  
            return cxt, _query, _candidates, (tgt, _dtgt, _dans0, *a, ans0, _ans), label
        return (_gen(True), _gen(False)) if counter_paired else _gen(random() < 0.5)
    wrapped.__name__ = f'g2c[{fn2str(g_fn)}]'
    return wrapped

def g2c(task, counter_paired=False):
    vocab_fn, gen_fn, inst, cxt2str, query2str, *a = task

    if not isinstance(cxt2str, partial):  # tasks_r, remove_local_hop
        def new_query2str(q, v): return 'Answer with Yes or No. ' + capitalize(query2str(q, v))
        task = (vocab_fn, _g2c(gen_fn, counter_paired=counter_paired), inst, cxt2str, new_query2str, *a)
        return task
    
    new_cxt2str = deepcopy(cxt2str)
    new_cxt2str.keywords['prefix'] = 'Premise: < '  # cxt > 1
    # new_cxt2str.keywords['prefix'], new_cxt2str.keywords['suffix'] = 'Premise: ', ''  # cxt == 1
    
    def new_query2str(q, v):
        s = query2str(q, v)
        if s.startswith('But '):
            s = s.replace('But ', '')
            s = try_wh_question2statement(s, v[1])
            # return 'Answer with Yes or No. ' + s
            return 'Answer with Yes or No. Is it possible that ' + s
            # return 'Answer with Yes, No or Unknown. Is it true that ' + s
            # return 'Answer with Yes or No. Does the premise contradict the statement that ' + s
        else:
            s = s.replace('So ', '')
            s = try_wh_question2statement(s, v[1])
            return 'Answer with Yes or No. Can it be inferred from the premise that ' + s
            # return 'Answer with No or Maybe. Can it be inferred from the premise that ' + s  # 0.53 0.75 / 0.89 0.625
            # return 'Answer with Yes, No or Maybe. So is it likely that ' + s  # 0.52 0.68 / 0.60 0.66
            # return 'Answer with No or Maybe. So may it be possible that ' + s  # better
            # return 'Answer with Yes or No. So is it possible that ' + s  # better
            # return 'Answer with No or Maybe. So can it be true that ' + s  # 0.43 0.75 / 0.55 0.718
            # return 'Answer with No or Maybe. Given the premise, can it be true that ' + s
            # return 'Answer with No or Maybe. So ' + s
            # return 'Answer with No or Maybe. So is it true that ' + s  #  / 0.60 0.718
            # return 'Answer with No or Maybe. So, ' + s  0.43 0.68 / 0.51 0.625
    
    cls_labels=['No', 'Yes'] if query2str('Q', vocab_fn()).startswith('But ') else ['Yes', 'No']
    task = (vocab_fn, _g2c(gen_fn, cls_labels=cls_labels, counter_paired=counter_paired),
            inst, new_cxt2str, new_query2str, *a)
    return task

def corrupt_query(task):
    vocab_fn, gen_fn, inst, cxt2str, query2str, *a = task

    def new_gen_fn(*args,**kwargs):
        cxt, query, candidates, (tgt, *a, ans0, ans), *label = gen_fn(*args,**kwargs)
        vocabs = args[0]
        rel0, rel1 = [v.relations[0] for v in vocabs]
        new_query = choice(list_diff(rel0.dom(), candidates2dict(candidates)['query']))
        return (cxt, query, candidates, (tgt, *a, ans0, ans), *label), \
            (cxt, new_query, candidates, (tgt, *a, ans0, ans), *label)
    new_gen_fn.__name__ = f'corrupt_query[{fn2str(gen_fn)}]'

    task = vocab_fn, new_gen_fn, inst, cxt2str, query2str, *a
    return task

def has_local_hop(task):
    vocab_fn, *a = task; vocabs = vocab_fn()
    return vocabs[0].data != vocabs[1].data

def transform_and_validate_task(task, rel0_i=None, rel1_i=None,
                rel0_kwargs=None, rel1_kwargs=None, do_swap_qa=False, do_negate=False,
                do_rm_query=False, do_g2c=False, do_corrupt_query=False,
                cxt_len=3, rev_item2str=False, abstract=False):
    args = {k: v for k, v in locals().items() if k not in ['task', 'e']}
    try:
        if rel0_i is not None: task = choose_rels(task, [rel0_i, rel1_i])
        if task is None: return None
        if rel0_kwargs is not None: task = decorate_rel(task, 0, rel0_kwargs)
        if rel1_kwargs is not None: task = decorate_rel(task, 1, rel1_kwargs)
        # if not has_local_hop(task) and do_swap_qa:
        #     raise InvalidTransException("invalid transformation rm_local_hop + swap_qa")
        if do_swap_qa: task = swap_qa(task)
        task = refine_query2str(task, do_swap_qa=do_swap_qa)
        if do_negate: task = negate(task)
        if not has_local_hop(task): task = remove_local_hop(task, do_swap_qa, do_rm_query, do_g2c, cxt_len)
        if do_rm_query: task = remove_query(task)
        if do_g2c: task = g2c(task, counter_paired=do_g2c == 'counter_paired')
        if do_corrupt_query: task = corrupt_query(task)
    except InvalidTransException as e:
        print(f'\ntransform_task failed: {e} ({args2str(args)})')
        return None
        
    vocab_fn, gen_fn, *_ = task
    rels = [vocab.relations[0] for vocab in vocab_fn()]
    if not has_local_hop(task) and rev_item2str:
        print(f'\ninvalid args for rm_local_hop and rev_item2str: {args2str(args)}')
        return None
    if do_rm_query and cxt_len < 3:
        print(f'\ninvalid args for do_rm_query: cxt_len = {cxt_len}')
        return None
    # if cxt_len == 1 and (rels[1].name == 'equal' or rels[0].name != 'equal' or do_negate or do_g2c):
    #     print(f'\ninvalid args for cxt_len 1: {args2str(args)}')
    #     return None
    if rels[1].name == 'sibling' and not do_g2c and task[1].__name__ not in ['rlrlr_gen']:
        print(f'\ninvalid args for sibling: {args2str(args)}')
        return None
    return task

def make_data_tuple(text, examples, tokenizer, k_shot=3, bos_tokens=None, eos_tokens=None, s2s=False):
    #if isinstance(tokenizer, LLAMATokenizer): text = text.replace('\n', '\ \n') 
    use_api = isinstance(tokenizer, types.FunctionType)
    if not use_api: input_ids = tokenizer.encode(text, return_tensors='pt')
    else: input_ids = text
    # elif not tokenizer.is_chat: input_ids = text
    # else: input_ids, answer = split_answer(text, bos_tokens[-1])
    example_strs = text.strip('\n').split(NEW_LINE)  # strip the trailing '\n'
    if len(example_strs) == len(examples): instruction = None
    else: assert len(example_strs) == len(examples) + 1; instruction, *example_strs = example_strs
    ranges, bos_indices = None, None
    if not use_api:
        ranges = locate_ranges(examples, example_strs, tokenizer, input_ids[0].tolist(), bos_tokens, instruction=instruction)
        bos_indices = [r.bos[-1] - 1 for r in ranges]  # [r.bos[0] for r in ranges]
    bos_indices, eos_indices, answers, labels = locate_answers(input_ids, tokenizer,
        bos_indices=bos_indices, bos_tokens=bos_tokens, eos_tokens=eos_tokens, nrows=len(examples))
    if s2s:  # for t5 models
        bos_i, eos_i = bos_indices[-1], eos_indices[-1]
        assert eos_i == input_ids.size(1) - 1, f'{eos_i} != {input_ids.size()}[1] - 1'
        assert tokenizer.convert_ids_to_tokens(input_ids[0, -1].item()) == eos_tokens == '</s>', \
            f"{tokenizer.convert_ids_to_tokens(input_ids[0, -1].item())} != '</s>'"
        input_ids = torch.cat([input_ids[:, : bos_i + 1], input_ids[:, -1:]], dim=1) # append trailing '</s>'
        answers, labels = answers[-1:], labels[:, bos_i: eos_i - 1]
        bos_indices, eos_indices = [bos_i - bos_i], [eos_i - bos_i]
    elif not use_api:
        labels[:, :bos_indices[k_shot]] = -100  

    candidates, answer_indices = None, None
    if isinstance(examples[0], dict):  
        answer_indices = [0 for _ in ranges]
        # def get_id(r, name): return input_ids[0][getattr(r, name)[0]].item()
        # candidates = [[get_id(r, name) for name in ['ans0', 's1']] for r in ranges]  # ioi task
        candidates = [[tokenizer.encode(i)[1] for i in e['candidates'][-1]] for e in examples]  # wino task
        return input_ids, labels, ranges, example_strs, bos_indices, eos_indices, answers, candidates, answer_indices
    
    cxt, query, cands, *_ = examples[0]
    cands = candidates2dict(cands)
    if cands is not None and len(list(cands.values())[-1]) > 1:  # cxt_len > 1
        if use_api:
            prefix = ' ' if answers[-1][0].startswith(' ') else ''
            candidates = [[prefix + token for token in list(candidates2dict(cands).values())[-1]]
                        for cxt, query, cands, *_ in examples]
        else:
            prefix, encode = ('', partial(tokenizer.encode, add_special_tokens=False)) \
                if isinstance(tokenizer, LlamaTokenizer) else (' ', partial(tokenizer.encode))
            candidates = [[encode(prefix + token)[0] for token in list(candidates2dict(cands).values())[-1]]
                        for cxt, query, cands, *_ in examples]
        answer_indices = [get_answer_index(e) for e in examples]
    return input_ids, labels, ranges, example_strs, bos_indices, eos_indices, answers, candidates, answer_indices