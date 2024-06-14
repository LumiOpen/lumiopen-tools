#!/bin/bash

model_langs=(
    "gpt2:eng"
    "bigscience/bloom:fra"
    "bigscience/bloom:spa"
    "bigscience/bloom:deu"
    "TurkuNLP/gpt3-finnish-small:fin"
    "LumiOpen/Poro:fin"
    "LumiOpen/Poro:eng"
    "AI-Sweden-Models/gpt-sw3-126m:swe"
    "AI-Sweden-Models/gpt-sw3-126m:nob"
    "AI-Sweden-Models/gpt-sw3-126m:dan"
    "AI-Sweden-Models/gpt-sw3-126m:isl"
    "PlanTL-GOB-ES/gpt2-base-bne:spa"
)

for model_lang in "${model_langs[@]}"; do
    model="${model_lang%%:*}"
    lang="${model_lang#*:}"
    size=$(python3 vocab_size.py $model 2>/dev/null)
    result=$(
	python3 fertility.py $model flores200_dataset/dev/$lang* 2>/dev/null \
	    | perl -pe 's/^fertility\s+//; s/ /\t/; s/[()]//g'
    )
    echo "$model"$'\t'"$size"$'\t'"$lang"$'\t'"$result"
done
