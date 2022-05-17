from pathlib import Path

import penman
import torch

from mspring_bmr import ROOT
from mspring_bmr.evaluation import predict_amrs, compute_smatch, compute_smatch_graphs
from mspring_bmr.penman import encode
from mspring_bmr.utils import instantiate_loader, instantiate_model_and_tokenizer
import random
import numpy as np

random.seed(12)
np.random.seed(12)
torch.manual_seed(12)
torch.cuda.manual_seed_all(12)

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
        "--pred-path", type=Path, default=ROOT / "data/tmp/inf-pred.txt", help="Where to write predictions."
    )
    parser.add_argument(
        "--gold-path", type=Path, default=ROOT / "data/tmp/inf-gold.txt", help="Where to write the gold file."
    )
    parser.add_argument(
        "--sorted-pred-path", type=Path, default=ROOT / "data/tmp/inf-sorted-pred.txt", help="Where to write the sorted prediction file."
    )
    parser.add_argument(
        "--use-recategorization",
        action="store_true",
        help="Predict using Zhang recategorization on top of our linearization (requires recategorized sentences in input).",
    )
    parser.add_argument(
        "--penman-linearization", action="store_true", help="Predict using PENMAN linearization instead of ours."
    )
    parser.add_argument("--use-pointer-tokens", action="store_true")
    parser.add_argument("--raw-graph", action="store_true")
    parser.add_argument("--restore-name-ops", action="store_true")
    parser.add_argument("--return-all", action="store_true")
    parser.add_argument("--mode", type=str, default="amr", help="Mode used to run the program, either amr/amr+/bmr/amr2bmr.")
    parser.add_argument("--language", type=str, default="en_XX", help="Language of the target sentences.")

    args = parser.parse_args()
    device = torch.device(args.device)
    model, tokenizer, snt_tokenizer = instantiate_model_and_tokenizer(
        args.model,
        dropout=0.0,
        attention_dropout=0.0,
        penman_linearization=args.penman_linearization,
        use_pointer_tokens=args.use_pointer_tokens,
        raw_graph=args.raw_graph,
        mode=args.mode,
        language=args.language,
        direction="graph",
    )

    model.amr_mode = True

    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'], strict=False)
    model.tie_weights()

    model.to(device)

    gold_path = args.gold_path
    pred_path = args.pred_path
    sorted_pred_path = args.sorted_pred_path

    loader = instantiate_loader(
        args.datasets,
        tokenizer,
        snt_tokenizer,
        batch_size=args.batch_size,
        evaluation=True,
        out=gold_path,
        use_recategorization=args.use_recategorization,
    )
    loader.device = device

    decoder_token_id = 0 if args.model == "facebook/bart-large" else tokenizer.convert_tokens_to_ids("en_XX")

    graphs = predict_amrs(
        loader,
        model,
        tokenizer,
        beam_size=args.beam_size,
        decoder_start_token_id=decoder_token_id,
        restore_name_ops=args.restore_name_ops,
        return_all=args.return_all,
    )
    if args.return_all:
        graphs = [g for gg in graphs for g in gg]

    pieces = [encode(g) for g in graphs]
    pred_path.write_text("\n\n".join(pieces))

    if not args.return_all:
        score = compute_smatch_graphs(graphs, gold_path, pred_path, sorted_pred_path)
        print(f'Smatch: {score:.3f}')
