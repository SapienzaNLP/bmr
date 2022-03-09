from pathlib import Path

import penman
import torch

from mspring_bmr import ROOT
from mspring_bmr.evaluation import predict_amrs, compute_smatch, predict_sentences, predict_sentences_multilingual
from mspring_bmr.penman import encode
from mspring_bmr.utils import instantiate_loader, instantiate_loader_graphs, instantiate_model_and_tokenizer
from tqdm import tqdm
from mspring_bmr.dataset import reverse_direction
from mspring_bmr.IO import read_amr_data


def spring_predict_sentences_from_graph_list(
    dataset,
    checkpoint,
    mode="amr",
    language="en_XX",
    model_name="facebook/bart-large",
    beam_size=5,
    batch_size=500,
    device="cuda",
    use_reverse_decoder=False,
    deinvert=True,
    penman_linearization=True,
    collapse_name_ops=False,
    use_pointer_tokens=True,
    raw_graph=False,
    return_all=False,
):
    device = torch.device(device)
    model, tokenizer, snt_tokenizer = instantiate_model_and_tokenizer(
        model_name,
        checkpoint=checkpoint,
        dropout=0.0,
        attention_dropout=0.0,
        penman_linearization=penman_linearization,
        use_pointer_tokens=use_pointer_tokens,
        collapse_name_ops=collapse_name_ops,
        init_reverse=use_reverse_decoder,
        raw_graph=raw_graph,
        mode=mode,
        language=language,
        direction="text",
    )
    model.to(device)
    model.rev.amr_mode = False

    loader = instantiate_loader_graphs(dataset, tokenizer, snt_tokenizer, batch_size=batch_size)
    loader.device = device

    shuffle_orig = loader.shuffle
    sort_orig = loader.sort
    total = len(loader.dataset)

    loader.shuffle = False
    loader.sort = True

    model.eval()
    model.amr_mode = False

    ids = []
    tokens = []

    with tqdm(total=total) as bar:
        for x, y, extra in tqdm(loader):
            ids.extend(extra["ids"])
            x, y = reverse_direction(x, y)
            x["input_ids"] = x["input_ids"][:, :1024]
            x["attention_mask"] = x["attention_mask"][:, :1024]
            with torch.no_grad():
                out = model.generate(
                    **x,
                    max_length=350,
                    decoder_start_token_id=0,
                    num_beams=beam_size,
                    num_return_sequences=beam_size,
                    early_stopping=False,
                    no_repeat_ngram_size=4,
                    length_penalty=0
                )

            for i1 in range(0, len(out), beam_size):
                tokens_same_source = []
                tokens.append(tokens_same_source)
                for i2 in range(i1, i1 + beam_size):
                    tokk = out[i2]
                    tokk = [t for t in tokk.tolist() if t > 2]
                    tokens_same_source.append(tokk)

                bar.update(out.size(0) // beam_size)

            # reorder
            tokens = [tokens[i] for i in ids]

    sentences = []
    for tokens_same_source in tokens:
        if return_all:
            sentences.append([tokenizer.decode(tokk).strip() for tokk in tokens_same_source])
        else:
            sentences.append(tokenizer.decode(tokens_same_source[0]).strip())

    loader.shuffle = shuffle_orig
    loader.sort = sort_orig

    return sentences


def spring_predict_amr_from_amr_file(
    datasets,
    checkpoint,
    mode="amr",
    language="en_XX",
    model_name="facebook/bart-large",
    beam_size=5,
    batch_size=500,
    device="cuda",
    use_recategorization=True,
    restore_name_ops=True,
    penman_linearization=True,
    use_pointer_tokens=True,
    raw_graph=False,
    return_all=False,
):


    device = torch.device(device)
    model, tokenizer, snt_tokenizer = instantiate_model_and_tokenizer(
        model_name,
        checkpoint=checkpoint,
        dropout=0.0,
        attention_dropout=0.0,
        penman_linearization=penman_linearization,
        use_pointer_tokens=use_pointer_tokens,
        raw_graph=raw_graph,
        mode=mode,
        language=language,
        direction="graph",
    )

    model.amr_mode = True
    model.to(device)

    loader = instantiate_loader(
        datasets,
        tokenizer,
        batch_size=batch_size,
        evaluation=False,
        use_recategorization=use_recategorization,
    )
    loader.device = device

    decoder_token_id = 0 if model_name == "facebook/bart-large" else tokenizer.convert_tokens_to_ids("en_XX")


    graphs = predict_amrs(
        loader,
        model,
        tokenizer,
        beam_size=beam_size,
        decoder_start_token_id=decoder_token_id,
        restore_name_ops=restore_name_ops,
        return_all=return_all,
    )
    if return_all:
        graphs = [g for gg in graphs for g in gg]

    pieces = [encode(g) for g in graphs]
    return pieces


def read_file_in_batches(sentences, batch_size=1000, max_length=100):

    data = []
    idx = 0
    for line in sentences:
        line = line.strip()
        idx += 1
        if not line:
            continue
        n = len(line.split())
        if n > max_length:
            continue
        data.append((idx, line, n))

    def _iterator(data):

        data = sorted(data, key=lambda x: x[2], reverse=True)

        maxn = 0
        batch = []

        for sample in data:
            idx, line, n = sample
            if n > batch_size:
                if batch:
                    yield batch
                    maxn = 0
                    batch = []
                yield [sample]
            else:
                curr_batch_size = maxn * len(batch)
                cand_batch_size = max(maxn, n) * (len(batch) + 1)

                if 0 < curr_batch_size <= batch_size and cand_batch_size > batch_size:
                    yield batch
                    maxn = 0
                    batch = []
                maxn = max(maxn, n)
                batch.append(sample)

        if batch:
            yield batch

    return _iterator(data), len(data)


def spring_predict_amr_from_sentence_list(
    sentences_list,
    checkpoint,
    mode="amr",
    language="en_XX",
    model_name="facebook/bart-large",
    beam_size=5,
    batch_size=500,
    device="cuda",
    use_recategorization=True,
    restore_name_ops=True,
    penman_linearization=True,
    use_pointer_tokens=True,
    raw_graph=False,
    only_ok=True,
):
    device = torch.device(device)
    model, tokenizer, snt_tokenizer = instantiate_model_and_tokenizer(
        model_name,
        checkpoint=checkpoint,
        dropout=0.0,
        attention_dropout=0,
        penman_linearization=penman_linearization,
        use_pointer_tokens=use_pointer_tokens,
        mode=mode,
        language=language,
        direction="graph",
    )
    model.to(device)
    model.eval()

    iterator, nsent = read_file_in_batches(sentences_list, batch_size)

    decoder_start_token_id = 0 if model_name == "facebook/bart-large" else snt_tokenizer.convert_tokens_to_ids("en_XX")

    for batch in tqdm(iterator):
        if not batch:
            continue
        ids, sentences, _ = zip(*batch)

        x = snt_tokenizer.batch_encode_plus(list(sentences), return_tensors='pt', padding=True)
        x = {k: v.to(device) for k, v in x.items()}

        with torch.no_grad():
            model.amr_mode = True
            out = model.generate(**x, max_length=batch_size, decoder_start_token_id=decoder_start_token_id, num_beams=beam_size)


        bgraphs = []
        for idx, sent, tokk in zip(ids, sentences, out):
            graph, status, (lin, backr) = tokenizer.decode_amr(tokk.tolist(), restore_name_ops=restore_name_ops)
            if only_ok and ("OK" not in str(status)):
                continue

            # graph.metadata['status'] = str(status)
            graph.metadata["id"] = str(idx)
            graph.metadata["lng"] = language
            graph.metadata["snt"] = sent
            bgraphs.append([idx, graph])

    bgraphs.sort(key=lambda x: x[0])

    graphs_predictions_graphs = [encode(g) for (_, g) in bgraphs]
    return graphs_predictions_graphs


def read_frames_in_batches(pb_frames_dict, batch_size=1000, max_length=100):
    data = []
    idx = 0
    for key, value in pb_frames_dict.items():
        line = value["text"].strip()
        idx += 1
        if not line:
            continue
        n = len(line.split())
        if n > max_length:
            continue

        arguments = []
        # all values distinct to text
        for k, v in value.items():
            if k == "text":
                continue
            arguments.append((k, v))

        data.append((idx, line, n, key, arguments))

    def _iterator(data):

        data = sorted(data, key=lambda x: x[2], reverse=True)

        maxn = 0
        batch = []

        for sample in data:
            idx, line, n, key, arguments = sample
            if n > batch_size:
                if batch:
                    yield batch
                    maxn = 0
                    batch = []
                yield [sample]
            else:
                curr_batch_size = maxn * len(batch)
                cand_batch_size = max(maxn, n) * (len(batch) + 1)

                if 0 < curr_batch_size <= batch_size and cand_batch_size > batch_size:
                    yield batch
                    maxn = 0
                    batch = []
                maxn = max(maxn, n)
                batch.append(sample)

        if batch:
            yield batch

    return _iterator(data), len(data)


def spring_predict_amr_from_pb_frame(
    pb_frames_dict,
    checkpoint,
    mode="amr",
    language="en_XX",
    model_name="facebook/bart-large",
    beam_size=5,
    batch_size=500,
    device="cuda",
    use_recategorization=True,
    restore_name_ops=True,
    penman_linearization=True,
    use_pointer_tokens=True,
    raw_graph=False,
    only_ok=True,
):
    device = torch.device(device)
    model, tokenizer, snt_tokenizer = instantiate_model_and_tokenizer(
        model_name,
        checkpoint=checkpoint,
        dropout=0.0,
        attention_dropout=0,
        penman_linearization=penman_linearization,
        use_pointer_tokens=use_pointer_tokens,
        mode=mode,
        language=language,
        direction="graph",
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    model.to(device)
    model.eval()

    iterator, nsent = read_frames_in_batches(pb_frames_dict, batch_size)
    bgraphs = []
    decoder_start_token_id = 0 if model_name == "facebook/bart-large" else snt_tokenizer.convert_tokens_to_ids("en_XX")

    for batch in tqdm(iterator):
        if not batch:
            continue
        ids, sentences, _, key, arguments = zip(*batch)
        x = snt_tokenizer.batch_encode_plus(list(sentences), return_tensors='pt', padding=True)

        with torch.no_grad():
            model.amr_mode = True
            out = model.generate(**x, max_length=512, decoder_start_token_id=decoder_start_token_id, num_beams=beam_size)

        for idx, sent, tokk, frame, argument in zip(ids, sentences, out, key, arguments):
            graph, status, (lin, backr) = tokenizer.decode_amr(tokk.tolist(), restore_name_ops=restore_name_ops)
            if only_ok and ("OK" not in str(status)):
                continue

            # graph.metadata['status'] = str(status)
            graph.metadata["id"] = str(idx)
            graph.metadata["snt"] = sent
            graph.metadata["pb-frame"] = frame
            for k, v in argument:
                if k == "rel":
                    graph.metadata["rel"] = v
                else:
                    graph.metadata["arg-" + k] = v
            bgraphs.append([idx, graph])

    bgraphs.sort(key=lambda x: x[0])

    graphs_predictions_graphs = [encode(g) for (_, g) in bgraphs]

    return graphs_predictions_graphs


def write_graphs(graphs, filename):
    with open(filename, "w") as f:
        for g in graphs:
            f.write(g + "\n\n")
