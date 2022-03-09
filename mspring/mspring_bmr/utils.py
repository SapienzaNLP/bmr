from glob import glob
from importlib.resources import path
from pathlib import Path

import torch
from transformers import AutoConfig

from mspring_bmr.dataset import AMRDataset, AMRDatasetTokenBatcherAndLoader
from mspring_bmr.modeling_bart import AMRBartForConditionalGeneration
from mspring_bmr.tokenization_bart import (
    AMRBartTokenizer,
    PENMANBartTokenizer,
    BMRBartTokenizer,
    BMRPENMANBartTokenizer,
)
from mspring_bmr.modeling_mbart import AMRMBartForConditionalGeneration
from mspring_bmr.tokenization_mbart import (
    AMRMBartTokenizer,
    PENMANMBartTokenizer,
    BMRMBartTokenizer,
    BMRPENMANMBartTokenizer,
)

from transformers.models.mbart.tokenization_mbart_fast import *
from transformers.models.bart.tokenization_bart_fast import *

def instantiate_model_and_tokenizer(
        name='facebook/bart-large',
        checkpoint=None,
        additional_tokens_smart_init=True,
        dropout = 0.15,
        attention_dropout = 0.15,
        from_pretrained = True,
        init_reverse = False,
        collapse_name_ops = False,
        penman_linearization = False,
        use_pointer_tokens = False,
        raw_graph = False,
        direction = "graph",
        mode = "bmr",
        language = "en_XX",
):
    if raw_graph:
        assert penman_linearization

    skip_relations = False

    tokenizer_name = name

    config = AutoConfig.from_pretrained(name)
    config.output_past = False
    config.no_repeat_ngram_size = 0
    config.prefix = " "
    config.output_attentions = True
    config.dropout = dropout
    config.attention_dropout = attention_dropout

    tokenizer_type = None
    snt_tokenizer_type = None
    model_type = None

    if penman_linearization and mode == "amr" and name =='facebook/bart-large':
        snt_tokenizer_type = BartTokenizerFast
        tokenizer_type = PENMANBartTokenizer
        model_type = AMRBartForConditionalGeneration

    elif penman_linearization and mode == "amr":
        snt_tokenizer_type = MBartTokenizerFast
        tokenizer_type = PENMANMBartTokenizer
        model_type = AMRMBartForConditionalGeneration

    elif penman_linearization and name =='facebook/bart-large':
        snt_tokenizer_type = BartTokenizerFast
        tokenizer_type = BMRPENMANBartTokenizer
        model_type = AMRBartForConditionalGeneration

    elif penman_linearization:         
        snt_tokenizer_type = MBartTokenizerFast
        tokenizer_type = BMRPENMANMBartTokenizer
        model_type = AMRMBartForConditionalGeneration

    elif mode == "amr" and name =='facebook/bart-large':
        snt_tokenizer_type = BartTokenizerFast
        tokenizer_type = AMRBartTokenizer
        model_type = AMRBartForConditionalGeneration

    elif mode == "amr":
        snt_tokenizer_type = MBartTokenizerFast
        tokenizer_type = AMRMBartTokenizer
        model_type = AMRMBartForConditionalGeneration

    elif name =='facebook/bart-large':
        snt_tokenizer_type = BartTokenizerFast
        tokenizer_type = BMRBartTokenizer
        model_type = AMRBartForConditionalGeneration

    else:
        snt_tokenizer_type = MBartTokenizerFast
        tokenizer_type = BMRMBartTokenizer
        model_type = AMRMBartForConditionalGeneration


    src_lang=language
    tgt_lang="en_XX"


    if penman_linearization:
        tokenizer = tokenizer_type.from_pretrained(
            tokenizer_name,
            collapse_name_ops=collapse_name_ops,
            use_pointer_tokens=use_pointer_tokens,
            raw_graph=raw_graph,
            config=config,
            direction=direction,
            mode=mode,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            add_prefix_space=True,
        )
    else:
        tokenizer = tokenizer_type.from_pretrained(
            tokenizer_name,
            collapse_name_ops=collapse_name_ops,
            use_pointer_tokens=use_pointer_tokens,
            config=config,
            direction=direction,
            mode=mode,
            src_lang=src_lang, 
            tgt_lang=tgt_lang,
            add_prefix_space=True,
        )

    snt_tokenizer = snt_tokenizer_type.from_pretrained(            
            tokenizer_name,
            collapse_name_ops=collapse_name_ops,
            use_pointer_tokens=use_pointer_tokens,
            config=config,
            src_lang=src_lang, 
            tgt_lang=tgt_lang,
            add_prefix_space=True,
    )

    model = model_type.from_pretrained(name, config=config) if from_pretrained else model_type(config)
    model.resize_token_embeddings(len(tokenizer))

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'], strict=False)
    else:
        if additional_tokens_smart_init:
            modified = 0
            for tok in tokenizer.added_tokens_list:
                idx = tokenizer.convert_tokens_to_ids(tok)

                tok = tok.lstrip(tokenizer.INIT)

                if idx < tokenizer.vocab_size:
                    continue

                elif tok.startswith("<pointer:") and tok.endswith(">"):
                    tok_split = ["pointer", str(tok.split(":")[1].strip(">"))]

                elif tok.startswith("<"):
                    continue

                elif tok.startswith(":"):

                    if skip_relations:
                        continue

                    elif tok.startswith(":op"):
                        tok_split = ["relation", "operator", str(int(tok[3:]))]

                    elif tok.startswith(":snt"):
                        tok_split = ["relation", "sentence", str(int(tok[4:]))]

                    elif tok.startswith(":ARG"):
                        tok_split = ["relation", "argument", str(int(tok[4:]))]

                    elif mode == "amr":
                        tok_split = ["relation"] + tok.lstrip(":").split("-")

                    else:
                        tok_split = ["relation"] + tok.lstrip(":").split("_")

                else:
                    tok_split = tok.split("-")

                tok_split_ = tok_split
                tok_split = []
                for s in tok_split_:
                    s_ = s + tokenizer.INIT
                    if tokenizer.unk_token != s_ and tokenizer.convert_tokens_to_ids(s_) != tokenizer.unk_token_id:
                        tok_split.append(s_)
                    else:
                        tok_split.extend(tokenizer._tok_bpe(s))

                vecs = []
                for s in tok_split:
                    idx_split = tokenizer.convert_tokens_to_ids(s)
                    if idx_split != tokenizer.unk_token_id:
                        vec_split = model.model.shared.weight.data[idx_split].clone()
                        vecs.append(vec_split)

                if vecs:
                    vec = torch.stack(vecs, 0).mean(0)
                    noise = torch.empty_like(vec)
                    noise.uniform_(-0.1, +0.1)
                    model.model.shared.weight.data[idx] = vec + noise
                    modified += 1

            if mode == "bmr":
                bn_lemmas_map = {}
                with open(f"./data/lemmas/lemmas_{language[:2].upper()}.tsv", "r") as f0:
                    for line in f0:
                        bn_lemmas_map["_" + line.strip().split("\t")[0][3:]] = (
                            line.strip().split("\t")[1][1:-1].split(", ")[:1]
                        )

                for bn, lemmas in bn_lemmas_map.items():
                    idx = tokenizer.convert_tokens_to_ids(bn)
                    if idx != tokenizer.unk_token_id:
                        tok = tok.lstrip(tokenizer.INIT)

                        tok_split_ = lemmas
                        vecs = []
                        for s in tok_split_:
                            s_ = tokenizer.convert_tokens_to_ids(tokenizer.INIT + s)
                            if s_ != tokenizer.unk_token_id:
                                vec_split = model.model.shared.weight.data[s_].clone()
                                vecs.append(vec_split)
                            else:
                                word_tokens = []

                                vec_word = []
                                for word in s.split("_"):
                                    word_ = tokenizer.convert_tokens_to_ids(tokenizer.INIT + s)

                                    if word_ != tokenizer.unk_token_id:
                                        vec_split = model.model.shared.weight.data[word_].clone()
                                        vec_word.append(vec_split)
                                    else:
                                        vec_word_tok = []
                                        for word_tok in tokenizer._tok_bpe(word):
                                            word_tok_ = tokenizer.convert_tokens_to_ids(word_tok)
                                            if word_tok_ != tokenizer.unk_token_id:
                                                vec_split = model.model.shared.weight.data[word_tok_].clone()
                                                vec_word_tok.append(vec_split)

                                        vec_word.append(torch.stack(vec_word_tok, 0).mean(0))

                                vecs.append(torch.stack(vec_word, 0).mean(0))

                        if vecs:
                            vec = torch.stack(vecs, 0).mean(0)
                            noise = torch.empty_like(vec)
                            noise.uniform_(-0.1, +0.1)
                            model.model.shared.weight.data[idx] = vec + noise
                            modified += 1

                del bn_lemmas_map

        model.model.set_input_embeddings(model.model.shared)
        if init_reverse:
            model.init_reverse_model()

    return model, tokenizer, snt_tokenizer


def instantiate_loader(
        glob_pattn,
        tokenizer,
        snt_tokenizer,
        batch_size=500,
        evaluation=True,
        out=None,
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
        raw_data=True
):
    if raw_data:
        paths = []
        if isinstance(glob_pattn, str) or isinstance(glob_pattn, Path):
            glob_pattn = [glob_pattn]
        for gpattn in glob_pattn:
            paths += [Path(p) for p in glob(gpattn)]
        if evaluation:
            assert out is not None
            Path(out).write_text("\n\n".join([p.read_text() for p in paths]))
    else:
        paths = glob_pattn

    dataset = AMRDataset(
        paths,
        tokenizer,
        snt_tokenizer,
        use_recategorization=use_recategorization,
        remove_longer_than=remove_longer_than,
        remove_wiki=remove_wiki,
        dereify=dereify,
        raw_data=raw_data,
    )
    loader = AMRDatasetTokenBatcherAndLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not evaluation,
    )
    return loader


def instantiate_loader_graphs(
        graphs,
        tokenizer,
        snt_tokenizer,
        batch_size=500,
        evaluation=True,
        out=None,
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
):

    dataset = AMRDataset(
        graphs,
        tokenizer,
        snt_tokenizer,
        use_recategorization=use_recategorization,
        remove_longer_than=remove_longer_than,
        remove_wiki=remove_wiki,
        dereify=dereify,
        raw_data=False,
    )
    loader = AMRDatasetTokenBatcherAndLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not evaluation,
    )
    return loader
