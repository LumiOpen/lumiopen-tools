## Tokenizer fertility evaluation

Compute `token count`/`word count` for given tokenizer and text document.

### Quickstart

Download and unpack [FLORES-200](https://github.com/facebookresearch/flores/tree/main/flores200) dataset

```
wget https://tinyurl.com/flores200dataset -O flores200_dataset.tar.gz
tar xzf flores200_dataset.tar.gz
```

Run `fertility.py` using the `gpt2` tokenizer and the FLORES-200 English
dev data

```
python3 fertility.py gpt2 flores200_dataset/dev/eng_Latn.dev 
fertility 26738/24274 (1.10)
```

Run `fertility.py` for various combinations of tokenizer and dataset
(requires FLORES-200 to be downloaded as above)

```
run.sh
```
