########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys
import numpy as np
import tensorflow as tf


class PIPELINE:
    def __init__(self, WORD_NAME):
        if WORD_NAME == 'cl100k_base':
            import tiktoken
            self.tokenizer = tiktoken.get_encoding(WORD_NAME)
        elif WORD_NAME == 'rwkv_vocab_v20230424':
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from rwkv_tokenizer import TRIE_TOKENIZER
            self.tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt')        
        else:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(WORD_NAME)

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    def encode(self, x):
        if 'Tokenizer' in str(type(self.tokenizer)):
            return self.tokenizer.encode(x).ids
        else:
            return self.tokenizer.encode(x)
    
    def decode(self, x):
        return self.tokenizer.decode(x)

    def sample_logits(self, probs, temperature=1.0, top_p=0.85, top_k=0):
        top_k = int(top_k)
        sorted_ids = np.argsort(probs)
        sorted_probs = probs[sorted_ids][::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        probs = probs / np.sum(probs)
        out = np.random.choice(a=len(probs), p=probs)
        return int(out)
