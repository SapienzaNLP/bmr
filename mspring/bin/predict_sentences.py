from pathlib import Path

import penman
import torch

from mspring_bmr import ROOT
from mspring_bmr.evaluation import (
    predict_amrs,
    compute_smatch,
    predict_sentences,
    compute_bleu,
    predict_sentences_multilingual,
)
from mspring_bmr.penman import encode
from mspring_bmr.utils import instantiate_loader, instantiate_model_and_tokenizer

if __name__ == "__main__":

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Script to predict AMR graphs given sentences. LDC format as input.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        nargs="+",
        help="Required. One or more glob patterns to use to load amr files.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Required. Checkpoint to restore.")
    parser.add_argument(
        "--model", type=str, default="facebook/bart-large", help="Model config to use to load the model class."
    )
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size.")
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size (as number of linearized graph tokens per batch)."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    parser.add_argument(
        "--pred-path", type=Path, default=ROOT / "data/tmp/inf-pred-sentences.txt", help="Where to write predictions."
    )
    parser.add_argument(
        "--gold-path", type=Path, default=ROOT / "data/tmp/inf-gold-sentences.txt", help="Where to write the gold file."
    )
    parser.add_argument("--add-to-graph-file", action="store_true")
    parser.add_argument("--use-reverse-decoder", action="store_true")
    parser.add_argument("--deinvert", action="store_true")
    parser.add_argument(
        "--penman-linearization", action="store_true", help="Predict using PENMAN linearization instead of ours."
    )
    parser.add_argument("--collapse-name-ops", action="store_true")
    parser.add_argument("--use-pointer-tokens", action="store_true")
    parser.add_argument("--raw-graph", action="store_true")
    parser.add_argument("--return-all", action="store_true")
    parser.add_argument("--mode", type=str, default="bmr", help="Mode used to run the program, either amr/amr+/bmr/amr2bmr.")
    parser.add_argument("--language", type=str, default="en_XX", help="Language of the target sentences.")
    args = parser.parse_args()


    device = torch.device(args.device)
    model, tokenizer, snt_tokenizer = instantiate_model_and_tokenizer(
        args.model,
        dropout=0.0,
        attention_dropout=0.0,
        penman_linearization=args.penman_linearization,
        use_pointer_tokens=args.use_pointer_tokens,
        collapse_name_ops=args.collapse_name_ops,
        init_reverse=args.use_reverse_decoder,
        raw_graph=args.raw_graph,
        mode=args.mode,
        language=args.language,
        direction="text",
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])

    model.to(device)
    model.rev.amr_mode = False

    loader = instantiate_loader(
        args.datasets, tokenizer, snt_tokenizer, batch_size=args.batch_size, evaluation=True, out="/tmp/a.txt", dereify=args.deinvert
    )

    loader.device = device

    if args.model == "facebook/bart-large":
        decoder_start_token_id = 0
    else:
        decoder_token_id = tokenizer.convert_tokens_to_ids(args.language)
        
    pred_sentences = predict_sentences_multilingual(
        loader, model.rev, tokenizer, beam_size=args.beam_size, return_all=args.return_all
    )
    if args.add_to_graph_file:
        graphs = loader.dataset.graphs
        for ss, g in zip(pred_sentences, graphs):
            if args.return_all:
                g.metadata["snt-pred"] = "\t\t".join(ss)
            else:
                g.metadata["snt-pred"] = ss
        args.pred_path.write_text("\n\n".join([encode(g) for g in graphs]))
    else:
        if args.return_all:
            pred_sentences = [s for ss in pred_sentences for s in ss]

        predictions = []
        gold_sentences = []
        for idx, g in enumerate(loader.dataset.graphs):
            # gold_sentences.append(g.metadata["id"] + "\t" + loader.dataset.sentences[idx])
            # predictions.append(g.metadata["id"] + "\t" + pred_sentences[idx])
            gold_sentences.append(loader.dataset.sentences[idx])
            predictions.append(pred_sentences[idx])


        args.gold_path.write_text("\n".join(gold_sentences))
        args.pred_path.write_text("\n".join(predictions))

        if not args.return_all:
            score = compute_bleu(loader.dataset.sentences, pred_sentences)
            print(f"BLEU: {score.score:.2f}")
