from pathlib import Path

import torch

try:
    from torch.cuda.amp import autocast

    autocast_available = True
except ImportError:

    class autocast:
        def __init__(self, enabled=True):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
            pass

    autocast_available = False

from torch.cuda.amp.grad_scaler import GradScaler
import transformers

from mspring_bmr import ROOT
from mspring_bmr.dataset import reverse_direction
from mspring_bmr.optim import RAdam
from mspring_bmr.evaluation import (
    write_predictions,
    compute_smatch,
    predict_amrs,
    predict_sentences,
    compute_bleu,
    predict_sentences_multilingual,
)
from mspring_bmr.utils import instantiate_model_and_tokenizer, instantiate_loader

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint, global_step_from_engine

import time
import random
import numpy as np

random.seed(112)
np.random.seed(112)
torch.manual_seed(112)
torch.cuda.manual_seed_all(112)


def do_train(checkpoint=None, model_name="facebook/bart-base", direction="graph", mode="amr", language="en_XX", split_both_decoder=False, fp16=False):

    assert direction in ("graph", "text", "both")
    start = time.time()

    model, tokenizer, snt_tokenizer = instantiate_model_and_tokenizer(
        model_name,
        checkpoint=checkpoint,
        additional_tokens_smart_init=config["smart_init"],
        dropout=config["dropout"],
        attention_dropout=config["attention_dropout"],
        from_pretrained=config["warm_start"],
        init_reverse=split_both_decoder,
        penman_linearization=config['penman_linearization'],
        collapse_name_ops=config['collapse_name_ops'],
        use_pointer_tokens=config['use_pointer_tokens'],
        raw_graph=config.get('raw_graph', False),
        direction=direction,
        mode=mode,
        language=language,
    )
    print(model)
    print(model.config)

    if checkpoint is not None:
        print(f"Checkpoint restored ({checkpoint})!")

    if direction == "both" and split_both_decoder:
        params_dir_enc = list(model.model.encoder.parameters())
        params_dir_enc_check = {id(p) for p in params_dir_enc}
        params_dir_dec = set()
        params_dir_dec |= {p for p in model.model.decoder.parameters() if id(p) not in params_dir_enc_check}
        params_dir_dec |= {p for p in model.rev.model.decoder.parameters() if id(p) not in params_dir_enc_check}
        params_dir_dec = list(params_dir_dec)
        optimizer = RAdam(
            [
                {"params": params_dir_enc, "lr": config["learning_rate"]},
                {"params": params_dir_dec, "lr": config["learning_rate"] * 2},
            ],
            weight_decay=config["weight_decay"],
        )
    else:
        optimizer = RAdam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    if checkpoint is not None:
        optimizer.load_state_dict(torch.load(checkpoint)["optimizer"])

    if config["scheduler"] == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=config["warmup_steps"], num_training_steps=config["training_steps"]
        )
    elif config["scheduler"] == "constant":
        scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config["warmup_steps"])
    else:
        raise ValueError

    scaler = GradScaler(enabled=fp16)

    train_loader = instantiate_loader(
        config["train"],
        tokenizer,
        snt_tokenizer,
        batch_size=config["batch_size"],
        evaluation=False,
        use_recategorization=config["use_recategorization"],
        remove_longer_than=config["remove_longer_than"],
        remove_wiki=config["remove_wiki"],
        dereify=config["dereify"],
    )

    dev_gold_path = ROOT / "data/tmp/dev-gold.txt"
    dev_pred_path = ROOT / "data/tmp/dev-pred.txt"
    dev_loader = instantiate_loader(
        config["dev"],
        tokenizer,
        snt_tokenizer,
        batch_size=config["batch_size"],
        evaluation=True,
        out=dev_gold_path,
        use_recategorization=config["use_recategorization"],
        remove_wiki=config["remove_wiki"],
        dereify=config["dereify"],
    )

    if direction == "graph":
        def train_step(engine, batch):
            model.train()
            x, y, extra = batch
            model.amr_mode = True

            with autocast(enabled=fp16):
                output = model(**x, **y)


            loss = output["loss"]
            scaler.scale((loss / config["accum_steps"])).backward()
            return loss.item()

        @torch.no_grad()
        def eval_step(engine, batch):
            model.eval()
            x, y, extra = batch
            model.amr_mode = True
            output = model(**x, **y)

            loss = output["loss"]
            return loss.item()

    elif direction == "text":

        def train_step(engine, batch):
            model.train()
            x, y, extra = batch
            x, y = reverse_direction(x, y)
            model.rev.amr_mode = False

            with autocast(enabled=fp16):
                output = model.rev(**x, **y)

            loss = output["loss"]
            scaler.scale((loss / config["accum_steps"])).backward()
            return loss.item()

        @torch.no_grad()
        def eval_step(engine, batch):
            model.eval()
            x, y, extra = batch
            x, y = reverse_direction(x, y)
            model.rev.amr_mode = False
            output = model(**x, **y)
            loss = output["loss"]
            return loss.item()

    elif direction == "both":

        def train_step(engine, batch):
            model.train()
            x, y, extra = batch
            model.amr_mode = True
            with autocast(enabled=fp16):
                output1 = model(**x, **y)

            loss1 = output1["loss"]

            scaler.scale((loss1 / config["accum_steps"] * 0.5)).backward()
            loss1 = loss1.item()
            x, y = reverse_direction(x, y)
            model.rev.amr_mode = False
            with autocast(enabled=fp16):
                output2 = model.rev(**x, **y)

            loss2 = output["loss"]
            scaler.scale((loss2 / config["accum_steps"] * 0.5)).backward()
            return loss1, loss2.item()

        @torch.no_grad()
        def eval_step(engine, batch):
            model.eval()
            x, y, extra = batch
            model.amr_mode = True
            output1 = model(**x, **y)
            loss1 = output["loss"]
            x, y = reverse_direction(x, y)
            output2 = model(**x, **y)
            loss2 = output["loss"]
            loss2, *_ = model.rev(**x, **y)
            return loss1.item(), loss2.item()

    else:
        raise ValueError

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    @trainer.on(Events.STARTED)
    def update(engine):
        print("training started!")

    @trainer.on(Events.EPOCH_COMPLETED)
    @trainer.on(Events.ITERATION_COMPLETED(every=config["accum_steps"]))
    def update(engine):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_trn_loss(engine):
        log_msg = f"training epoch: {engine.state.epoch}"
        if direction in ("graph", "both"):
            log_msg += f" | loss_amr: {engine.state.metrics['trn_amr_loss']:.3f}"
        if direction in ("text", "both"):
            log_msg += f" | loss_text: {engine.state.metrics['trn_text_loss']:.3f}"
        print(log_msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_dev_eval(engine):
        dev_loader.batch_size = config["batch_size"]
        dev_loader.device = next(model.parameters()).device
        evaluator.run(dev_loader)

    if not config["best_loss"]:
        if direction in ("graph", "both"):

            @evaluator.on(Events.EPOCH_COMPLETED)
            def smatch_eval(engine):
                device = next(model.parameters()).device
                dev_loader.device = device

                if model_name == "facebook/bart-base":
                    decoder_token = 0
                else:
                    decoder_token = tokenizer.convert_tokens_to_ids("en_EN")

                graphs = predict_amrs(
                    dev_loader,
                    model,
                    tokenizer,
                    decoder_start_token_id=decoder_token,
                    restore_name_ops=config["collapse_name_ops"],
                )
                write_predictions(dev_pred_path, tokenizer, graphs)
                try:
                    smatch = compute_smatch(dev_gold_path, dev_pred_path)
                except:
                    smatch = 0.0
                engine.state.metrics["dev_smatch"] = smatch

        if direction in ("text", "both"):

            @evaluator.on(Events.EPOCH_COMPLETED)
            def smatch_eval(engine):
                device = next(model.parameters()).device
                dev_loader.device = device


                if model_name == "facebook/bart-base":
                    decoder_token = 0
                else:
                    decoder_token = tokenizer.convert_tokens_to_ids(language)

                pred_sentences = predict_sentences_multilingual(
                    dev_loader,
                    model.rev,
                    tokenizer,
                    decoder_start_token_id=decoder_token,
                    beam_size=config["beam_size"],
                )
                Path(dev_pred_path).write_text("\n".join(pred_sentences))
                bleu = compute_bleu(dev_loader.dataset.sentences, pred_sentences)
                engine.state.metrics["dev_bleu"] = bleu.score

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_dev_loss(engine):
        log_msg = f"dev epoch: {trainer.state.epoch}"
        if direction in ("graph", "both"):
            log_msg += f" | loss_amr: {engine.state.metrics['dev_amr_loss']:.3f}"
            if not config["best_loss"]:
                log_msg += f" | smatch: {engine.state.metrics['dev_smatch']:.3f}"
        if direction in ("text", "both"):
            log_msg += f" | loss_text: {engine.state.metrics['dev_text_loss']:.3f}"
            if not config["best_loss"]:
                log_msg += f" | bleu: {engine.state.metrics['dev_bleu']:.3f}"
        print(log_msg)

    if direction == "graph":
        RunningAverage(output_transform=lambda out: out).attach(trainer, "trn_amr_loss")
        RunningAverage(output_transform=lambda out: out).attach(evaluator, "dev_amr_loss")
    elif direction == "text":
        RunningAverage(output_transform=lambda out: out).attach(trainer, "trn_text_loss")
        RunningAverage(output_transform=lambda out: out).attach(evaluator, "dev_text_loss")
    elif direction == "both":
        RunningAverage(output_transform=lambda out: out[0]).attach(trainer, "trn_amr_loss")
        RunningAverage(output_transform=lambda out: out[1]).attach(trainer, "trn_text_loss")
        RunningAverage(output_transform=lambda out: out[0]).attach(evaluator, "dev_amr_loss")
        RunningAverage(output_transform=lambda out: out[1]).attach(evaluator, "dev_text_loss")

    if config["log_wandb"]:
        from ignite.contrib.handlers.wandb_logger import WandBLogger

        wandb_logger = WandBLogger(init=False)

        if direction == "graph":
            wandb_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="iterations/trn_amr_loss",
                output_transform=lambda loss: loss,
            )
        elif direction == "text":
            wandb_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="iterations/trn_text_loss",
                output_transform=lambda loss: loss,
            )
        if direction == "both":
            wandb_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="iterations/trn_amr_loss",
                output_transform=lambda loss: loss[0],
            )
            wandb_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="iterations/trn_text_loss",
                output_transform=lambda loss: loss[1],
            )

        if direction == "graph":
            metric_names_trn = ["trn_amr_loss"]
            metric_names_dev = ["dev_amr_loss"]
            if not config["best_loss"]:
                metric_names_dev.append("dev_smatch")
        elif direction == "text":
            metric_names_trn = ["trn_text_loss"]
            metric_names_dev = ["dev_text_loss"]
            if not config["best_loss"]:
                metric_names_dev.append("dev_bleu")
        elif direction == "both":
            metric_names_trn = ["trn_amr_loss", "trn_text_loss"]
            metric_names_dev = ["dev_amr_loss", "dev_smatch"]
            if not config["best_loss"]:
                metric_names_dev.extend(["dev_text_loss", "dev_bleu"])

        wandb_logger.attach_output_handler(
            trainer,
            event_name=Events.EPOCH_COMPLETED,
            tag="epochs",
            metric_names=metric_names_trn,
            global_step_transform=lambda *_: trainer.state.iteration,
        )

        wandb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="epochs",
            metric_names=metric_names_dev,
            global_step_transform=lambda *_: trainer.state.iteration,
        )

        @trainer.on(Events.ITERATION_COMPLETED)
        def wandb_log_lr(engine):
            wandb.log({"lr": scheduler.get_last_lr()[0]}, step=engine.state.iteration)

    if config["save_checkpoints"]:

        if direction in ("graph", "both"):
            if config["best_loss"]:
                prefix = "best-loss-amr"
                score_function = lambda x: 1 / evaluator.state.metrics["dev_amr_loss"]
            else:
                prefix = "best-smatch"
                score_function = lambda x: evaluator.state.metrics["dev_smatch"]
        else:
            if config["best_loss"]:
                prefix = "best-loss-text"
                score_function = lambda x: 1 / evaluator.state.metrics["dev_amr_loss"]
            else:
                prefix = "best-bleu"
                score_function = lambda x: evaluator.state.metrics["dev_bleu"]

        to_save = {"model": model, "optimizer": optimizer}
        if config["log_wandb"]:
            where_checkpoints = str(wandb_logger.run.dir)
        else:
            root = ROOT / "runs"
            try:
                root.mkdir()
            except:
                pass
            where_checkpoints = root / str(len(list(root.iterdir())))
            try:
                where_checkpoints.mkdir()
            except:
                pass
            where_checkpoints = str(where_checkpoints)

        print(where_checkpoints)
        handler = ModelCheckpoint(
            where_checkpoints,
            prefix,
            n_saved=1,
            create_dir=True,
            score_function=score_function,
            global_step_transform=global_step_from_engine(trainer),
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler, to_save)

    model.cuda()
    device = next(model.parameters()).device
    train_loader.device = device
    trainer.run(train_loader, max_epochs=config["max_epochs"])


if __name__ == "__main__":

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import yaml

    import wandb

    parser = ArgumentParser(
        description="Trainer script",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="graph",
        choices=["graph", "text", "both"],
        help="Train a uni- (graph, text) or bidirectional (both).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="amr",
        choices=["amr", "amr+", "bmr", "bmr*"],
        help="Train a model with (amr, amr+, bmr, bmr*).",
    )
    parser.add_argument(
        "--model", type=str, default="facebook/bart-base", help="Model config to use to load the model class."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en_XX",
        help="Train in a specific language.",
    )
    parser.add_argument("--split-both-decoder", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/sweeped.yaml", help="Use the following config for hparams."
    )
    parser.add_argument("--checkpoint", type=str, help="Warm-start from a previous fine-tuned checkpoint.")
    parser.add_argument("--fp16", action="store_true")
    
    args, unknown = parser.parse_known_args()

    if args.fp16 and autocast_available:
        raise ValueError("You'll need a newer PyTorch version to enable fp16 training.")

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    if config["log_wandb"]:
        wandb.init(entity="carlosml26", project="spring-test", config=config, dir=str(ROOT / "runs/splits/"))
        config = wandb.config

    print(config)
    
    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = None

    do_train(
        checkpoint=checkpoint,
        model_name=args.model,
        direction=args.direction,
        mode=args.mode,
        language=args.language,
        split_both_decoder=args.split_both_decoder,
        fp16=args.fp16,
    )
