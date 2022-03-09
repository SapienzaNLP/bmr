## English AMR+ parsing
## Use:
## - data/vocab/bmr/special_tokens-amr.txt
## - data/vocab/bmr/additions-old.txt
python3 bin/predict_amrs_from_plaintext.py \
    --texts ../bmr-parser/res/corpora/omw_glosses/preprocessed/glosses_en.tsv \
    --checkpoint models/parsing/AMR+-en-smatch_0.8209.pt \
    --beam-size 5 \
    --batch-size 500 \
    --device cuda \
    --penman-linearization \
    --use-pointer-tokens \
    --mode amr+ \
    --language en_XX

## English BMR parsing
## Use:
## - data/vocab/bmr/special_tokens.txt
## - data/vocab/bmr/additions.txt
python3 bin/predict_amrs_from_plaintext.py \
    --texts ../bmr-parser/res/corpora/omw_glosses/preprocessed/glosses_en.tsv \
    --checkpoint models/parsing/BMR-en-smatch_0.7859.pt \
    --beam-size 5 \
    --batch-size 500 \
    --device cuda \
    --penman-linearization \
    --use-pointer-tokens \
    --mode bmr \
    --language en_XX