# In the bmr/filtered/ folder the AMR/AMR+/BMR/AMR2BMR graphs in the drive from the English
# sentences/filtered/ in ENGLISH

# predict_sentences: Language in italian it_IT and mode in BMR FOR ITALIAN
# amr -> mode: amr, vocab/amr/additions
#
# bmr -> mode: bmr, vocab/bmr/additions (specifies the structure), vocab/bmr/special_tokens (using synsets)
# amr+ -> mode: bmr, vocab/bmr/additions (specifies the structure), vocab/bmr/special_tokens (not using synsets)
#
# amr2bmr -> mode: bmr, vocab/bmr/additions (specifies the structure), vocab/bmr/special_tokens (using synsets)

# ----------------------------------------------------------------------

## Italian BMR generation (vocab/bmr/special_tokens points to a file with synsets)
python3 bin/predict_sentences.py \
    --datasets ../bmr-parser/res/bmr/filtered/bmr-en/graph-pred.text.txt \
    --gold-path ../bmr-parser/res/sentences/filtered/bmr-it/gold.text.txt \
    --pred-path ../bmr-parser/res/sentences/filtered/bmr-it/pred.text.txt \
    --checkpoint models/generation/BMR-it-bleu_36.9762.pt \
    --beam-size 5 \
    --batch-size 500 \
    --device cuda \
    --penman-linearization \
    --use-pointer-tokens \
    --model facebook/mbart-large-cc25 \
    --mode bmr \
    --language it_IT

## Italian AMR+ generation (vocab/bmr/special_tokens points to an empty file)
python3 bin/predict_sentences.py \
    --datasets ../bmr-parser/res/bmr/filtered/amr+-en/graph-pred.text.txt  \
    --gold-path ../bmr-parser/res/sentences/filtered/amr+-it/gold.text.txt \
    --pred-path ../bmr-parser/res/sentences/filtered/amr+-it/pred.text.txt \
    --checkpoint models/generation/AMR+-it-bleu_36.7262.pt \
    --beam-size 5 \
    --batch-size 500 \
    --device cuda \
    --penman-linearization \
    --use-pointer-tokens \
    --model facebook/mbart-large-cc25 \
    --mode amr+ \
    --language it_IT

## Italian BMR_FROM_AMR generation (vocab/bmr/special_tokens points to a file with synsets)
python3 bin/predict_sentences.py \
    --datasets ../bmr-parser/res/bmr/filtered/bmr-en-from_amr/pred_tok.amr.t0.8.v1-reduced.tsv  \
    --gold-path ../bmr-parser/res/sentences/filtered/bmr-it-from_amr/gold.text.txt \
    --pred-path ../bmr-parser/res/sentences/filtered/bmr-it-from_amr/pred.text.txt \
    --checkpoint models/generation/BMR-it-bleu_36.9762.pt \
    --beam-size 5 \
    --batch-size 500 \
    --device cuda \
    --penman-linearization \
    --use-pointer-tokens \
    --model facebook/mbart-large-cc25 \
    --mode amr2bmr \
    --language it_IT

## Italian AMR generation
python3 bin/predict_sentences.py \
    --datasets ../bmr-parser/res/bmr/filtered/amr-en/pred_tok.amr.t0.8.v1.txt  \
    --gold-path ../bmr-parser/res/sentences/filtered/amr-it/gold.text.txt \
    --pred-path ../bmr-parser/res/sentences/filtered/amr-it/pred.text.txt \
    --checkpoint models/generation/AMR-it-bleu_33.1816.pt \
    --beam-size 5 \
    --batch-size 500 \
    --device cuda \
    --penman-linearization \
    --use-pointer-tokens \
    --model facebook/mbart-large-cc25 \
    --mode amr \
    --language it_IT
