from pathlib import Path

import penman
import torch
from tqdm import tqdm

from mspring_bmr.penman import encode
from mspring_bmr.utils import instantiate_model_and_tokenizer


def read_file_in_batches_gloss(path, batch_size=1000, max_length=100):

    data = []
    idx = 0
    for line in Path(path).read_text().strip().splitlines():
        line = line.strip()
        idx += 1
        if not line:
            continue
        n = len(line.split())
        if n > max_length:
            continue
        synset_id, sentence_id, lemma, gloss = line.split("\t")
        gloss = gloss.strip()
        data.append((idx, gloss, n, lemma, synset_id, sentence_id))

    def _iterator(data):

        data = sorted(data, key=lambda x: x[2], reverse=True)

        maxn = 0
        batch = []

        for sample in data:
            idx, line, n, lemma, synset_id, sentence_id = sample
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

def read_file_in_batches(path, batch_size=1000, max_length=100):

    data = []
    idx = 0
    for line in Path(path).read_text().strip().splitlines():
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

if __name__ == "__main__":

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Script to predict AMR graphs given sentences. LDC format as input.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--texts",
        type=str,
        required=True,
        nargs="+",
        help="Required. One or more files containing \\n-separated sentences.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Required. Checkpoint to restore.")
    parser.add_argument(
        "--model", type=str, default="facebook/bart-large", help="Model config to use to load the model class."
    )
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size.")
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size (as number of linearized graph tokens per batch)."
    )
    parser.add_argument(
        "--penman-linearization", action="store_true", help="Predict using PENMAN linearization instead of ours."
    )
    parser.add_argument("--use-pointer-tokens", action="store_true")
    parser.add_argument("--restore-name-ops", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    parser.add_argument("--only-ok", action="store_true")
    parser.add_argument("--mode", type=str, default="bmr", help="Mode used to run the program, either amr or bmr.")
    parser.add_argument("--language", type=str, default="en_XX", help="Language of the source sentences.")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer, snt_tokenizer = instantiate_model_and_tokenizer(
        args.model,
        dropout=0.0,
        attention_dropout=0,
        penman_linearization=args.penman_linearization,
        use_pointer_tokens=args.use_pointer_tokens,
        mode=args.mode,
        language=args.language,
        direction="amr",
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.to(device)
    model.eval()

    graphs_predictions = []

    decoder_start_token_id = 0 if args.model == "facebook/bart-large" else snt_tokenizer.convert_tokens_to_ids(args.language)

    for path in tqdm(args.texts, desc="Files:"):

        iterator, nsent = read_file_in_batches(path, args.batch_size)
        with tqdm(desc=path, total=nsent) as bar:
            for batch in iterator:
                if not batch:
                    continue
                # ids, sentences, _, lemmas, synset_ids, sentence_ids = zip(*batch)
                # x, _ = tokenizer.batch_encode_sentences(sentences, device=device)
                ids, sentences, _ = zip(*batch)

                batch = snt_tokenizer.batch_encode_plus(list(sentences), return_tensors='pt', padding=True)
                batch["input_ids"][batch["input_ids"]==snt_tokenizer.convert_tokens_to_ids(snt_tokenizer.src_lang)] = decoder_start_token_id
                x = {k: v.to(device) for k, v in batch.items()}

                with torch.no_grad():
                    model.amr_mode = True
                    out = model.generate(**x, max_length=args.batch_size, decoder_start_token_id=snt_tokenizer.convert_tokens_to_ids(snt_tokenizer.src_lang), num_beams=args.beam_size)

                bgraphs = []
                # for idx, sent, tokk, lemma, synset_id, sentence_id in zip(ids, sentences, out, lemmas, synset_ids, sentence_ids):
                #     graph, status, (lin, backr) = tokenizer.decode_amr(tokk.tolist(), restore_name_ops=args.restore_name_ops)
                #     if args.only_ok and ("OK" not in str(status)):
                #         continue
                #     # graph.metadata['status'] = str(status)
                #     graph.metadata["source"] = path
                #     graph.metadata["nsent"] = str(idx)
                #     graph.metadata["snt"] = sent
                #     if lemma.strip() != "LEMMA_PLACEHOLDER":
                #         graph.metadata['lemma'] = lemma.strip()
                #     graph.metadata['synset_id'] = synset_id.strip()
                #     graph.metadata['id'] = sentence_id.strip()
                #     bgraphs.append((idx, graph))

                for idx, sent, tokk in zip(ids, sentences, out):
                    graph, status, (lin, backr) = tokenizer.decode_amr(tokk.tolist(), restore_name_ops=args.restore_name_ops)
                    if args.only_ok and ('OK' not in str(status)):
                        continue
                    #graph.metadata['status'] = str(status)
                    graph.metadata['source'] = path
                    graph.metadata['id'] = str(idx)
                    graph.metadata['snt'] = sent
                    bgraphs.append((idx, graph))
                for i, g in bgraphs:
                    graphs_predictions.append([i, g])

        graphs_predictions.sort(key=lambda x: x[0])
        graphs_predictions_graphs = [encode(g) for (_, g) in graphs_predictions]
        new_file = path.replace(".txt", "_bmr.txt")
        with open("graphs.txt", "w") as f0:
            f0.write("\n\n".join(graphs_predictions_graphs))

        exit(0)

        ids, graphs = zip(*sorted(results, key=lambda x: x[0]))

        for g in graphs:
            print(encode(g))
            print()
