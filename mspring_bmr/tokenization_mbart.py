import copy
import sys
from pathlib import Path
from tkinter import E

import penman
import regex as re
import torch
from transformers.models.mbart.tokenization_mbart_fast import *

from mspring_bmr import ROOT, postprocessing
from mspring_bmr.linearization import AMRTokens, AMRLinearizer
from mspring_bmr.penman import encode
from mspring_bmr.postprocessing import decode_into_node_and_backreferences

class AMRMBartTokenizer(MBartTokenizerFast):
    INIT = "▁" 

    ADDITIONAL = [
        AMRTokens.PNTR_N,
        AMRTokens.STOP_N,
        AMRTokens.LIT_START,
        AMRTokens.LIT_END,
        AMRTokens.BACKR_SRC_N,
        AMRTokens.BACKR_TRG_N,
    ]

    def __init__(self, *args, use_pointer_tokens=False, collapse_name_ops=False, direction="amr", mode="amr" , **kwargs):
        super().__init__(*args, **kwargs)
        self.patterns = re.compile(
            r""" ?<[a-z]+:?\d*>| ?:[^\s]+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.linearizer = AMRLinearizer(use_pointer_tokens=use_pointer_tokens, collapse_name_ops=collapse_name_ops)
        self.use_pointer_tokens = use_pointer_tokens
        self.collapse_name_ops = collapse_name_ops
        self.recategorizations = set()
        self.modified = 0
        self.direction = direction
        self.added_tokens_list = []
        self.special_tokens_list = [self.eos_token, self.pad_token, '<mask>', '<unk>']
        self.vocab_path = f'data/vocab/{mode}/'
        self.special_tokens_path = "special_tokens.txt"
        

    @classmethod
    def from_pretrained(cls, pretrained_model_path, pred_min=5, *args, **kwargs):
        inst = super().from_pretrained(pretrained_model_path, *args, **kwargs)
        inst.init_amr_vocabulary(pred_min=pred_min)
        return inst

    def init_amr_vocabulary(self, pred_min=5):
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        tokens = []
        for line in Path(ROOT / f"{self.vocab_path}predicates.txt").read_text().strip().splitlines():
            tok, count = line.split()
            if int(count) >= pred_min:
                tokens.append(tok)

        for tok in Path(ROOT / f"{self.vocab_path}additions.txt").read_text().strip().splitlines():
            tokens.append(tok)

        for tok in Path(ROOT / f"{self.vocab_path}recategorizations.txt").read_text().strip().splitlines():
            if not tok.startswith("_"):
                self.recategorizations.add(tok)
            tokens.append(tok)

        if self.use_pointer_tokens:
            for cnt in range(512):
                tokens.append(f"<pointer:{cnt}>")

        tokens += self.ADDITIONAL
        tokens = [self.INIT + t if t[0] not in ("_", "-") else t for t in tokens]
        tokens = [t for t in tokens if (self.unk_token != t and self.convert_tokens_to_ids(t) == self.unk_token_id)]
        self.added_tokens_list = tokens.copy()
        self.add_tokens(tokens)
        self.modified = len(tokens)



    def pre_tokenize(self, text):
        bpe_tokens = []
        for tok_span in text.lstrip().split(" "):
            tok_span = tok_span.strip()
            recats = tok_span.rsplit("_", 1)
            if (
                len(recats) == 2
                and recats[0] in self.recategorizations
                and (self.convert_tokens_to_ids("_" + recats[1]) != self.unk_token_id)
            ):
                bpe_tokens.extend([self.INIT + recats[0], "_" + recats[1]])

            else:
                for token in re.findall(self.pat, " " + tok_span):
                    bpe_tokens.extend(self.tokenize(token))
        return bpe_tokens

    def _tok_bpe(self, token):
        tokk = []
        tok = token.strip()
        recats = tok.rsplit("_", 1)
        if (
            len(recats) == 2
            and recats[0] in self.recategorizations
            and (self.convert_tokens_to_ids("_" + recats[1]) != self.unk_token_id)
        ):
            tokk.extend([self.INIT + recats[0], "_" + recats[1]])
        else:
            for tok in self.patterns.findall(" " + token):
                tokk.extend(self.tokenize(tok))

        return tokk

    def _get_nodes_and_backreferences(self, graph):
        lin = self.linearizer.linearize(graph)
        linearized_nodes, backreferences = lin.nodes, lin.backreferences
        return linearized_nodes, backreferences

    def tokenize_amr(self, graph):
        linearized_nodes, backreferences = self._get_nodes_and_backreferences(graph)
        bpe_tokens = []
        bpe_backreferences = []
        counter = 0


        for i, (backr, tokk) in enumerate(zip(backreferences, linearized_nodes)):
            is_in_enc = (self.unk_token != self.INIT + tokk and self.convert_tokens_to_ids(self.INIT + tokk) != self.unk_token_id) or (tokk in self.special_tokens_list)
            is_rel = tokk.startswith(':') and len(tokk) > 1
            is_spc = tokk.startswith('<') and tokk.endswith('>')
            is_of  = tokk.startswith(':') and tokk.endswith('-of')
            is_frame = re.match(r'.+-\d\d', tokk) is not None
            is_float = "." in tokk and tokk.split(".")[0].isnumeric() and tokk.split(".")[1].isnumeric()
            if tokk.startswith('"') and tokk.endswith('"'):
                tokk = tokk[1:-1].replace(" ","Ñ").replace('_', 'Ç')
                bpe_toks = [self.INIT + AMRTokens.LIT_START]
                bpe_toks += self._tok_bpe(tokk)
                bpe_toks.append(self.INIT + AMRTokens.LIT_END)

            elif is_rel or is_spc or is_frame or is_of:
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                elif is_frame:
                    bpe_toks = self._tok_bpe(tokk[:-3]) + [tokk[-3:]]
                elif is_of:
                    rel = tokk[:-3]
                    if (
                        self.unk_token != self.INIT + rel
                        and self.convert_tokens_to_ids(self.INIT + rel) != self.unk_token_id
                    ):
                        bpe_toks = [self.INIT + rel, "-of"]
                    else:
                        bpe_toks = [self.INIT + ':'] +  [tokkk if id_tokk == 0 else tokkk  for id_tokk, tokkk in enumerate(self._tok_bpe(rel[1:]))] + ['-of']
                elif is_rel:
                    bpe_toks = [self.INIT + ':'] + [tokkk if id_tokk == 0 else tokkk  for id_tokk, tokkk in enumerate(self._tok_bpe(tokk[1:]))]
                else:
                    raise

            else:
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                else:
                    bpe_toks = self._tok_bpe(tokk)
                    bpe_toks = bpe_toks[:1] + [tokkk for tokkk in bpe_toks[1:] if tokkk != self.INIT]

            bpe_tokens.append(bpe_toks)

            if i == backr:
                bpe_backr = list(range(counter, counter + len(bpe_toks)))
                counter += len(bpe_toks)
                bpe_backreferences.append(bpe_backr)
            else:
                bpe_backreferences.append(bpe_backreferences[backr][0:1])
                counter += 1
        bpe_tokens = [b for bb in bpe_tokens for b in bb]
        bpe_token_ids = [self.convert_tokens_to_ids(b) for b in bpe_tokens]
        bpe_backreferences = [b for bb in bpe_backreferences for b in bb]

        pre_tokenize_graph = " ".join(linearized_nodes)
        middle_tokens, backensss = decode_into_node_and_backreferences(bpe_token_ids, self)
        middle_tokens = [str(t) for t in middle_tokens]
        post_tokenize_graph = "( " + " ".join(middle_tokens)

        if pre_tokenize_graph.strip() !=  post_tokenize_graph.strip():
            print(bpe_tokens)
            print(graph.metadata['id'])
            print(pre_tokenize_graph)
            print(post_tokenize_graph)
            exit(0)
            
        return bpe_tokens, bpe_token_ids, bpe_backreferences

    def linearize(self, graph):
        shift = len(self)
        tokens, token_ids, backreferences = self.tokenize_amr(graph)

        tokens.insert(0, self.tgt_lang)
        token_ids.insert(0, self.convert_tokens_to_ids(self.tgt_lang))
        backreferences.append(len(backreferences))

        token_uni_ids = \
            [idx if i == b else b + shift for i, (idx, b) in enumerate(zip(token_ids, backreferences))]
        if token_uni_ids[-1] != (AMRTokens.EOS_N):
            tokens.append(AMRTokens.EOS_N)
            token_ids.append(self.eos_token_id)
            token_uni_ids.append(self.eos_token_id)
            backreferences.append(len(backreferences))

        if self.direction == "text":
            token_uni_ids = token_ids[1:] + token_ids[:1]
            tokens = tokens[1:] + tokens[:1]

        extra = {"linearized_graphs": tokens, "graphs": graph}

        return token_uni_ids, extra

    def batch_encode_graphs(self, graphs, device=torch.device("cpu")):
        linearized, extras = zip(*[self.linearize(g) for g in graphs])
        return self.batch_encode_graphs_from_linearized(linearized, extras, device=device)

    def batch_encode_graphs_from_linearized(self, linearized, extras=None, device=torch.device("cpu")):
        if extras is not None:
            batch_extra = {"linearized_graphs": [], "graphs": []}
            for extra in extras:
                batch_extra["graphs"].append(extra["graphs"])
                batch_extra["linearized_graphs"].append(extra["linearized_graphs"])
        else:
            batch_extra = {}
        maxlen = 0
        batch = []
        for token_uni_ids in linearized:
            maxlen = max(len(token_uni_ids), maxlen)
            batch.append(token_uni_ids)
        batch = [x + [self.pad_token_id] * (maxlen - len(x)) for x in batch]
        batch = torch.tensor(batch).to(device)
        batch = {"decoder_input_ids": batch[:, :-1], "labels": batch[:, 1:]}
        return batch, batch_extra

    def decode_amr(self, tokens, restore_name_ops=False):
        try:
            nodes, backreferences = postprocessing.decode_into_node_and_backreferences(tokens, self)
        except Exception as e:
            print("Decoding failure:", file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        if self.use_pointer_tokens:
            nodes, backreferences = postprocessing.restore_backreferences_from_pointers(nodes)
        try:
            graph_ = graph = postprocessing.build_graph(nodes, backreferences, restore_name_ops=restore_name_ops)
        except Exception as e:
            print("Building failure:", file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph, status = postprocessing.connect_graph_if_not_connected(graph)
            if status == postprocessing.ParsedStatus.BACKOFF:
                print("Reconnection 1 failure:")
                print(nodes, file=sys.stderr)
                print(backreferences, file=sys.stderr)
                print(graph_, file=sys.stderr)
            return graph, status, (nodes, backreferences)
        except Exception as e:
            print("Reconnction 2 failure:", file=sys.stderr)
            print(e, file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(graph_, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (nodes, backreferences)


class PENMANMBartTokenizer(AMRMBartTokenizer):
    def __init__(self, *args, raw_graph=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.linearizer = None
        self.remove_pars = False
        self.raw_graph = raw_graph

    def _tokenize_encoded_graph(self, encoded):
        linearized = re.sub(r"(\".+?\")", r" \1 ", encoded)
        pieces = []
        is_lit = False
        for piece in linearized.split():
            if piece.startswith('"') and piece.endswith('"'):
                pieces.append(piece)
            elif piece.startswith('"'):
                is_lit = True
                pieces.append(piece)

            elif is_lit:
                pieces[-1] +=  "@@" + piece
                is_lit = not piece.endswith('"')

            else:
                piece = piece.replace("(", " ( ")
                piece = piece.replace(")", " ) ")
                piece = piece.replace(":", " :")
                piece = piece.replace("/", " / ")
                piece = piece.strip()
                pieces.append(piece)
        linearized = re.sub(r'\s+', ' ', ' '.join(pieces)).strip()
        linearized_nodes = linearized.split(' ')
        linearized_nodes = [x.replace('@@', ' ') for x in linearized_nodes]
        return linearized_nodes

    def tokenize_amr(self, graph):
        if self.raw_graph:
            graph_ = copy.deepcopy(graph)
            graph_.metadata = {}
            linearized = penman.encode(graph_)
            linearized = re.sub(r"\s+", " ", linearized)
            bpe_tokens = self.pre_tokenize(linearized)[:1022]
            bpe_token_ids = [self.convert_tokens_to_ids(b) for b in bpe_tokens]
            bpe_backreferences = list(range(len(bpe_token_ids)))
            return bpe_tokens, bpe_token_ids, bpe_backreferences
        else:
            return super().tokenize_amr(graph)

    def _get_nodes_and_backreferences(self, graph):
        graph_ = copy.deepcopy(graph)
        graph_.metadata = {}
        linearized = penman.encode(graph_)
        linearized_nodes = self._tokenize_encoded_graph(linearized)

        if self.use_pointer_tokens:
            remap = {}
            for i in range(1, len(linearized_nodes)):
                nxt = linearized_nodes[i]
                lst = linearized_nodes[i - 1]
                if nxt == "/":
                    remap[lst] = f"<pointer:{len(remap)}>"
            i = 1
            linearized_nodes_ = [linearized_nodes[0]]
            while i < (len(linearized_nodes)):
                nxt = linearized_nodes[i]
                lst = linearized_nodes_[-1]
                if nxt in remap:
                    if lst == "(" and linearized_nodes[i + 1] == "/":
                        nxt = remap[nxt]
                        i += 1
                    elif lst.startswith(":"):
                        nxt = remap[nxt]
                linearized_nodes_.append(nxt)
                i += 1
            linearized_nodes = linearized_nodes_
            if self.remove_pars:
                linearized_nodes = [n for n in linearized_nodes if n != "("]

        backreferences = list(range(len(linearized_nodes)))
        return linearized_nodes, backreferences

    def _classify(self, node):
        if not isinstance(node, str):
            return "CONST"
        elif node == "i":
            return "I"
        elif re.match(r"^[a-z]\d*$", node) is not None:
            return "VAR"
        elif node[0].isdigit():
            return "CONST"
        elif node.startswith('"') and node.endswith('"'):
            return "CONST"
        elif node in ("+", "-"):
            return "CONST"
        elif node == ":mode":
            return "MODE"
        elif node.startswith(":"):
            return "EDGE"
        elif node in ["/", "(", ")"]:
            return node
        elif node[0].isalpha():
            for char in (",", ":", "/", "(", ")", ".", "!", "?", "\\"):
                if char in node:
                    return "CONST"
            return "INST"
        else:
            return "CONST"

    def _fix_and_make_graph(self, nodes):

        nodes_ = []
        for n in nodes:
            if isinstance(n, str):
                if n.startswith("<") and n.endswith(">") and (not n.startswith("<pointer:")):
                    pass
                else:
                    nodes_.append(n)
            else:
                nodes_.append(n)
        nodes = nodes_

        if self.use_pointer_tokens:

            i = 0
            nodes_ = []
            while i < len(nodes):
                nxt = nodes[i]
                pst = None
                if isinstance(nxt, str) and nxt.startswith("<pointer:"):
                    e = nxt.find(">")
                    if e != len(nxt) - 1:
                        pst = nxt[e + 1 :]
                        nxt = nxt[: e + 1]
                    nodes_.append(nxt)
                    if pst is not None:
                        nodes_.append(pst)
                else:
                    nodes_.append(nxt)
                i += 1
            nodes = nodes_

            i = 1
            nodes_ = [nodes[0]]
            while i < len(nodes):
                nxt = nodes[i]
                if isinstance(nxt, str) and nxt.startswith("<pointer:"):
                    nxt = "z" + nxt[9:-1]
                    fol = nodes[i + 1]
                    # is not expansion
                    if isinstance(fol, str) and (fol.startswith(":") or (fol == ")")):
                        nodes_.append(nxt)
                    else:
                        if self.remove_pars:
                            nodes_.append("(")
                        else:
                            if nodes_[-1] != "(":
                                nodes_.append("(")
                                # pass
                        nodes_.append(nxt)
                        nodes_.append("/")
                else:
                    nodes_.append(nxt)
                i += 1
            nodes = nodes_

        i = 0
        nodes_ = []
        while i < (len(nodes) - 1):
            if nodes[i] == ":":
                nodes_.append(nodes[i] + nodes[i + 1])
                i += 2
                last = False
            else:
                nodes_.append(nodes[i])
                i += 1
                last = True
        if last:
            nodes_.append(nodes[-1])
        nodes = nodes_

        i = 0
        nodes_ = []
        while i < (len(nodes)):
            if i < 2:
                nodes_.append(nodes[i])
                i += 1
            elif nodes_[-2] == "/" and nodes[i] == "/":
                i += 2
            else:
                nodes_.append(nodes[i])
                i += 1
        nodes = nodes_

        i = 0
        newvars = 0
        variables = set()
        remap = {}
        nodes_ = []
        while i < (len(nodes)):

            next = nodes[i]

            if next == "/":
                last = nodes_[-1]
                if last in variables:
                    last_remap = f"z{newvars+1000}"
                    newvars += 1
                    nodes_[-1] = last_remap
                    remap[last] = last_remap
                variables.add(last)
                nodes_.append(next)

            elif self._classify(next) == "VAR" and next in remap and (i < len(nodes) - 1) and nodes[i + 1] != "/":
                next = remap[next]
                nodes_.append(next)

            else:
                nodes_.append(next)

            i += 1

        nodes = nodes_
        pieces_ = []
        open_cnt = 0
        closed_cnt = 0
        if nodes[0] != "(":
            pieces_.append("(")
            open_cnt += 1
        for p in nodes:
            if p == "(":
                open_cnt += 1
            elif p == ")":
                closed_cnt += 1
            pieces_.append(p)
            if open_cnt == closed_cnt:
                break
        nodes = pieces_ + [")"] * (open_cnt - closed_cnt)

        pieces = []
        for piece in nodes:
            if not pieces:
                pieces.append("(")
            else:
                piece = str(piece)
                if piece.startswith('"') or piece.startswith('"') or '"' in piece.strip('"'):
                    piece = '"' + piece.replace('"', "") + '"'

                prev = self._classify(pieces[-1])
                next = self._classify(piece)

                if next == "CONST":
                    quote = False
                    for char in (",", ":", "/", "(", ")", ".", "!", "?", "\\", "_", "="):
                        if char in piece:
                            quote = True
                            break
                    if quote:
                        piece = '"' + piece.strip('"') + '"'

                if prev == "(":
                    if next in ("VAR", "I"):
                        pieces.append(piece)
                elif prev == ")":
                    if next in (")", "EDGE", "MODE"):
                        pieces.append(piece)
                elif prev == "VAR":
                    if next in ("/", "EDGE", "MODE", ")"):
                        pieces.append(piece)
                elif prev == "/":
                    if next in ("INST", "I"):
                        pieces.append(piece)
                elif prev == "INST":
                    if next in (")", "EDGE", "MODE"):
                        pieces.append(piece)
                elif prev == "I":
                    if next in ("/", ")", "EDGE", "MODE"):
                        pieces.append(piece)
                elif prev == "EDGE":
                    if next in ("(", "VAR", "CONST", "I"):
                        pieces.append(piece)
                    elif next == ")":
                        pieces[-1] = piece
                    elif next in ("EDGE", "MODE"):
                        pieces[-1] = piece
                elif prev == "MODE":
                    if next == "INST":
                        pieces.append(piece)
                elif prev == "CONST":
                    if next in (")", "EDGE", "MODE"):
                        pieces.append(piece)

        pieces_ = []
        open_cnt = 0
        closed_cnt = 0
        if pieces[0] != "(":
            pieces_.append("(")
            open_cnt += 1
        for p in pieces:
            if p == "(":
                open_cnt += 1
            elif p == ")":
                closed_cnt += 1
            pieces_.append(p)
            if open_cnt == closed_cnt:
                break
        pieces = pieces_ + [")"] * (open_cnt - closed_cnt)

        linearized = re.sub(r"\s+", " ", " ".join(pieces)).strip()

        """
        line = linearized
        # make sure parentheses match
        # copied from https://github.com/RikVN/AMR/blob/master/restoreAMR/restore_amr.py
        open_count = 0
        close_count = 0
        for i, c in enumerate(line):
            if c == '(':
                open_count += 1
            elif c == ')':
                close_count += 1
            if open_count == close_count and open_count > 0:
                line = line[:i].strip()
                break
        old_line = line
        while True:
            open_count = len(re.findall(r'\(', line))
            close_count = len(re.findall(r'\)', line))
            if open_count > close_count:
                line += ')' * (open_count - close_count)
            elif close_count > open_count:
                for i in range(close_count - open_count):
                    line = line.rstrip(')')
                    line = line.rstrip(' ')
            if old_line == line:
                break
            old_line = line
        """

        graph = penman.decode(linearized + " ")
        triples = []
        newvars = 2000
        for triple in graph.triples:
            x, rel, y = triple
            if x is None:
                pass
            elif rel == ":instance" and y is None:
                triples.append(penman.Triple(x, rel, "thing"))
            elif y is None:
                var = f"z{newvars}"
                newvars += 1
                triples.append(penman.Triple(x, rel, var))
                triples.append(penman.Triple(var, ":instance", "thing"))
            else:
                triples.append(triple)
        graph = penman.Graph(triples)
        linearized = encode(graph)

        def fix_text(linearized=linearized):
            n = 0

            def _repl1(match):
                nonlocal n
                out = match.group(1) + match.group(2) + str(3000 + n) + " / " + match.group(2) + match.group(3)
                n += 1
                return out

            linearized = re.sub(
                r"(\(\s?)([a-z])([^\/:\)]+[:\)])", _repl1, linearized, flags=re.IGNORECASE | re.MULTILINE
            )

            def _repl2(match):
                return match.group(1)

            linearized = re.sub(
                r"(\(\s*[a-z][\d+]\s*\/\s*[^\s\)\(:\/]+\s*)((?:/\s*[^\s\)\(:\/]+\s*)+)",
                _repl2,
                linearized,
                flags=re.IGNORECASE | re.MULTILINE,
            )

            # adds a ':' to args w/o it
            linearized = re.sub(r"([^:])(ARG)", r"\1 :\2", linearized)

            # removes edges with no node
            # linearized = re.sub(r':[^\s\)\(:\/]+?\s*\)', ')', linearized, flags=re.MULTILINE)

            return linearized

        linearized = fix_text(linearized)

        g = penman.decode(linearized)
        return g

    def decode_amr(self, tokens, restore_name_ops=None):
        try:
            if self.raw_graph:
                nodes = self._tokenize_encoded_graph(self.decode(tokens))
                backreferences = list(range(len(nodes)))
            else:
                nodes, backreferences = postprocessing.decode_into_node_and_backreferences(tokens, self)
            nodes_ = nodes
        except Exception as e:
            print("Decoding failure:", file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph_ = graph = self._fix_and_make_graph(nodes)
            if self.collapse_name_ops:
                graph_ = graph = postprocessing._split_name_ops(graph)
        except Exception as e:
            print("Building failure:", file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph, status = postprocessing.connect_graph_if_not_connected(graph)
            if status == postprocessing.ParsedStatus.BACKOFF:
                print("Reconnection 1 failure:")
                print(nodes, file=sys.stderr)
                print(backreferences, file=sys.stderr)
                print(graph_, file=sys.stderr)
            return graph, status, (nodes_, backreferences)
        except Exception as e:
            print("Reconnction 2 failure:", file=sys.stderr)
            print(e, file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(graph_, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (nodes_, backreferences)


class BMRMBartTokenizer(AMRMBartTokenizer):
    def init_amr_vocabulary(self, pred_min=5):
        super().init_amr_vocabulary(pred_min=pred_min)
        new_special_tokens = []
        for line in Path(ROOT / f"{self.vocab_path}{self.special_tokens_path}").read_text().strip().splitlines():
            stok, times = line.strip().split("\t")
            if int(times) > 1:
                new_special_tokens.append(stok)

        new_token = {"additional_special_tokens": new_special_tokens}

        # Add the synset as special token
        self.add_special_tokens(new_token)

    def tokenize_amr(self, graph):
        linearized_nodes, backreferences = self._get_nodes_and_backreferences(graph)
        bpe_tokens = []
        bpe_backreferences = []
        counter = 0

        for i, (backr, tokk) in enumerate(zip(backreferences, linearized_nodes)):
            is_in_enc = (
                self.unk_token != self.INIT + tokk and self.convert_tokens_to_ids(self.INIT + tokk) != self.unk_token_id
            ) or (tokk in self.special_tokens_list)
            is_rel = tokk.startswith(":") and len(tokk) > 1
            is_spc = tokk.startswith("<") and tokk.endswith(">")
            is_of = tokk.startswith(":") and tokk.endswith("_of")
            is_multi_word = "_" in tokk
            is_bn = "_" in tokk and tokk[-9:-1].isnumeric()

            if tokk.startswith('"') and tokk.endswith('"'):
                tokk = tokk[1:-1].replace(" ","Ñ").replace('_', 'Ç')
                bpe_toks = [self.INIT + AMRTokens.LIT_START]
                bpe_toks += self._tok_bpe(tokk)
                bpe_toks.append(self.INIT + AMRTokens.LIT_END)

            elif is_rel or is_spc or is_multi_word or is_of or is_bn:
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                elif is_of:
                    rel = tokk[:-3]
                    if (
                        self.unk_token != self.INIT + rel
                        and self.convert_tokens_to_ids(self.INIT + rel) != self.unk_token_id
                    ):
                        bpe_toks = [self.INIT + rel, "_of"]
                    else:
                        bpe_toks = [self.INIT + ':'] +  [tokkk if id_tokk == 0 else tokkk.replace(self.INIT, "")  for id_tokk, tokkk in enumerate(self._tok_bpe(rel[1:]))] + ['_of']
                elif is_rel:
                    bpe_toks = [self.INIT + ':'] + [tokkk if id_tokk == 0 else tokkk.replace(self.INIT, "")  for id_tokk, tokkk in enumerate(self._tok_bpe(tokk[1:]))]
                elif is_bn:
                    word_splitted = [self._tok_bpe(splitted_token) for splitted_token in tokk.split("_")[0:-1]]
                    word_splitted = [bb if idx_bb == 0 else ["_"] + bb for idx_bb, bb in enumerate(word_splitted)]
                    word_splitted = [b for bb in word_splitted for b in bb]
                    bpe_toks = word_splitted + [tokk[-10:]]
                elif is_multi_word:
                    word_splitted = [self._tok_bpe(splitted_token) for splitted_token in tokk.split("_")]
                    word_splitted = [bb if idx_bb == 0 else ["_"] + bb for idx_bb, bb in enumerate(word_splitted)]
                    bpe_toks = [b for bb in word_splitted for b in bb] 
                else:
                    raise

            else:
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                else:
                    bpe_toks = self._tok_bpe(tokk)
                    bpe_toks = bpe_toks[:1] + [tokkk for tokkk in bpe_toks[1:] if tokkk != self.INIT]

            bpe_tokens.append(bpe_toks)

            if i == backr:
                bpe_backr = list(range(counter, counter + len(bpe_toks)))
                counter += len(bpe_toks)
                bpe_backreferences.append(bpe_backr)
            else:
                bpe_backreferences.append(bpe_backreferences[backr][0:1])
                counter += 1
        bpe_tokens = [b for bb in bpe_tokens for b in bb]
        bpe_token_ids = [self.convert_tokens_to_ids(b) for b in bpe_tokens]
        bpe_backreferences = [b for bb in bpe_backreferences for b in bb]

        return bpe_tokens, bpe_token_ids, bpe_backreferences


class BMRPENMANMBartTokenizer(PENMANMBartTokenizer):
    def init_amr_vocabulary(self, pred_min=5):
        super().init_amr_vocabulary(pred_min=pred_min)
        new_special_tokens = []
        for line in Path(ROOT / f"{self.vocab_path}{self.special_tokens_path}").read_text().strip().splitlines():
            stok, times = line.strip().split("\t")
            if int(times) > 1:
                new_special_tokens.append(stok)

        new_token = {"additional_special_tokens": new_special_tokens}

        # Add the synset as special token
        self.add_special_tokens(new_token)

    def tokenize_amr(self, graph):
        if self.raw_graph:
            graph_ = copy.deepcopy(graph)
            graph_.metadata = {}
            linearized = penman.encode(graph_)
            linearized = re.sub(r"\s+", " ", linearized)
            bpe_tokens = self.pre_tokenize(linearized)[:1022]
            bpe_token_ids = [self.convert_tokens_to_ids(b) for b in bpe_tokens]
            bpe_backreferences = list(range(len(bpe_token_ids)))
            return bpe_tokens, bpe_token_ids, bpe_backreferences
        else:
            linearized_nodes, backreferences = self._get_nodes_and_backreferences(graph)
            bpe_tokens = []
            bpe_backreferences = []
            counter = 0

            for i, (backr, tokk) in enumerate(zip(backreferences, linearized_nodes)):
                is_in_enc = (self.unk_token != self.INIT + tokk and self.convert_tokens_to_ids(self.INIT + tokk) != self.unk_token_id)   or (tokk in self.special_tokens_list)
                is_rel = tokk.startswith(':') and len(tokk) > 1
                is_spc = tokk.startswith('<') and tokk.endswith('>')
                is_of  = tokk.startswith(':') and tokk.endswith('_of')
                is_frame = re.match(r'.+-\d\d', tokk) is not None
                is_multi_word = "_" in tokk
                is_bn = "_" in tokk and tokk[-9:-1].isnumeric()

                if tokk.startswith('"') and tokk.endswith('"'):
                    tokk = tokk[1:-1].replace(" ","Ñ").replace('_', 'Ç')
                    bpe_toks = [self.INIT + AMRTokens.LIT_START]
                    bpe_toks += self._tok_bpe(tokk)
                    bpe_toks.append(self.INIT + AMRTokens.LIT_END)

                elif (is_rel or is_spc or is_multi_word or is_of or is_bn or is_frame):
                    if is_in_enc:
                        bpe_toks = [self.INIT + tokk]
                    elif is_frame:
                        bpe_toks = self._tok_bpe(tokk[:-3]) + [tokk[-3:]]
                    elif is_of:
                        rel = tokk[:-3]
                        if (
                            self.unk_token != self.INIT + rel
                            and self.convert_tokens_to_ids(self.INIT + rel) != self.unk_token_id
                        ):
                            bpe_toks = [self.INIT + rel, "_of"]
                        else:
                            bpe_toks = [self.INIT + ':'] +  [tokkk if id_tokk == 0 else tokkk.replace(self.INIT, "")  for id_tokk, tokkk in enumerate(self._tok_bpe(rel[1:]))] + ['_of']
                    elif is_rel:
                        bpe_toks = [self.INIT + ':'] + [tokkk if id_tokk == 0 else tokkk.replace(self.INIT, "")  for id_tokk, tokkk in enumerate(self._tok_bpe(tokk[1:]))]
                    elif is_bn:
                        if tokk.startswith("bn_"):
                            word_splitted = [self.INIT + tokk.split("_")[0]]
                        else:
                            word_splitted = [self._tok_bpe(splitted_token) for splitted_token in tokk.split("_")[0:-1]]
                            word_splitted = [
                                bb if idx_bb == 0 else ["_"] + bb for idx_bb, bb in enumerate(word_splitted)
                            ]
                            word_splitted = [b for bb in word_splitted for b in bb]
                        bpe_toks = word_splitted + [tokk[-10:]]
                    elif is_multi_word:
                        word_splitted = [self._tok_bpe(splitted_token) for splitted_token in tokk.split("_")]
                        word_splitted = [bb if idx_bb == 0 else ["_"] + bb for idx_bb, bb in enumerate(word_splitted)]
                        bpe_toks = [b for bb in word_splitted for b in bb]  
                    else:
                        raise

                else:
                    if is_in_enc:
                        bpe_toks = [self.INIT + tokk]
                    else:
                        bpe_toks = self._tok_bpe(tokk)
                        bpe_toks = bpe_toks[:1] + [tokkk for tokkk in bpe_toks[1:] if tokkk != self.INIT]

                bpe_tokens.append(bpe_toks)

                if i == backr:
                    bpe_backr = list(range(counter, counter + len(bpe_toks)))
                    counter += len(bpe_toks)
                    bpe_backreferences.append(bpe_backr)
                else:
                    bpe_backreferences.append(bpe_backreferences[backr][0:1])
                    counter += 1
            bpe_tokens = [b for bb in bpe_tokens for b in bb]
            bpe_token_ids = [self.convert_tokens_to_ids(b) for b in bpe_tokens]
            bpe_backreferences = [b for bb in bpe_backreferences for b in bb]

            pre_tokenize_graph = " ".join(linearized_nodes)
            middle_tokens, backensss = decode_into_node_and_backreferences(bpe_token_ids, self)
            middle_tokens = [str(t) for t in middle_tokens]
            post_tokenize_graph = "( " + " ".join(middle_tokens)


            if pre_tokenize_graph.strip() !=  post_tokenize_graph.strip():
                print(bpe_tokens)
                print(graph.metadata['id'])
                print(pre_tokenize_graph)
                print(post_tokenize_graph)
                exit(0)

            return bpe_tokens, bpe_token_ids, bpe_backreferences
