The code for moving from AMR to BMR has sensitive data from AMR and BabelNet, therefore to obtain the BMR 1.0 please send us an email with a proof of the AMR license.

## Installation
```shell script
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
pip install -e .
```

The code only works with `transformers` 4.11. 
The code works fine with `torch` 1.5. We recommend the usage of a new `conda` env.

## Train
Modify `config.yaml` in `configs`. Instructions in comments within the file. Since the model is trained in a multilingual fashion, the AMR graph metadata has to include the field "# ::lng" with the language of the sentence (e.g., "en_XX", "es_XX", "de_DE", "it_IT", etc).

### Text-to-AMR
```shell script
python bin/train.py --config configs/config.yaml --direction graph --mode amr --model facebook/mbart-large-cc25
```
Results in `runs/`

### AMR-to-Text
```shell script
python bin/train.py --config configs/config.yaml --direction text --mode amr --model facebook/mbart-large-cc25
```
Results in `runs/`

## Evaluate
### Text-to-AMR
```shell script
python bin/predict_amrs.py \
    --datasets <AMR-ROOT>/data/amrs/split/test/*.txt \
    --gold-path data/tmp/amr3.0/gold.amr.txt \
    --pred-path data/tmp/amr3.0/pred.amr.txt \
    --checkpoint runs/<checkpoint>.pt \
    --beam-size 5 \
    --batch-size 500 \
    --device cuda \
    --penman-linearization --use-pointer-tokens
    --mode amr
    --model facebook/mbart-large-cc25
    --language en_XX

```
`gold.amr.txt` and `pred.amr.txt` will contain, respectively, the concatenated gold and the predictions.

To reproduce our paper's results, you will also need need to run the [BLINK](https://github.com/facebookresearch/BLINK) 
entity linking system on the prediction file (`data/tmp/amr3.0/pred.amr.txt` in the previous code snippet). 
To do so, you will need to install BLINK, and download their models:
```shell script
git clone https://github.com/facebookresearch/BLINK.git
cd BLINK
pip install -r requirements.txt
sh download_blink_models.sh
cd models
wget http://dl.fbaipublicfiles.com/BLINK//faiss_flat_index.pkl
cd ../..
```
Then, you will be able to launch the `blinkify.py` script:
```shell
python bin/blinkify.py \
    --datasets data/tmp/amr3.0/pred.amr.txt \
    --out data/tmp/amr3.0/pred.amr.blinkified.txt \
    --device cuda \
    --blink-models-dir BLINK/models
```
To have comparable Smatch scores you will also need to use the scripts available at https://github.com/mdtux89/amr-evaluation, which provide
results that are around ~0.3 Smatch points lower than those returned by `bin/predict_amrs.py`.

### AMR-to-Text
```shell script
python bin/predict_sentences.py \
    --datasets <AMR-ROOT>/data/amrs/split/test/*.txt \
    --gold-path data/tmp/amr3.0/gold.text.txt \
    --pred-path data/tmp/amr3.0/pred.text.txt \
    --checkpoint runs/<checkpoint>.pt \
    --beam-size 5 \
    --batch-size 500 \
    --device cuda \
    --penman-linearization --use-pointer-tokens
    --mode amr
    --model facebook/mbart-large-cc25
    --language en_XX
```
`gold.text.txt` and `pred.text.txt` will contain, respectively, the concatenated gold and the predictions.
For BLEU, chrF++, and Meteor in order to be comparable you will need to tokenize both gold and predictions using [JAMR tokenizer](https://github.com/redpony/cdec/blob/master/corpus/tokenize-anything.sh).
To compute BLEU and chrF++, please use `bin/eval_bleu.py`. For METEOR, use https://www.cs.cmu.edu/~alavie/METEOR/ .
For BLEURT don't use tokenization and run the eval with `https://github.com/google-research/bleurt`. Also see the [appendix](docs/appendix.pdf).

## Linearizations
The previously shown commands assume the use of the DFS-based linearization. To use BFS or PENMAN decomment the relevant lines in `configs/config.yaml` (for training). As for the evaluation scripts, substitute the `--penman-linearization --use-pointer-tokens` line with `--use-pointer-tokens` for BFS or with `--penman-linearization` for PENMAN.
