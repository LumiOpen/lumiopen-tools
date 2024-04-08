# lumiopen-tools

Tools for working with LumiOpen models

## Quickstart for LUMI

Start an interactive session on a GPU node

```
./slurm/gpu-sinteractive-puhti.sh
```

Set up environment

```
module use /appl/local/csc/modulefiles
module load pytorch
export HF_HOME=/scratch/project_462000319/hf_cache
```

Test generation with Poro

```
echo "The best life advice I've ever heard is this:" \
    | python3 generate.py LumiOpen/Poro-34B --dtype bf16 
```

Interactive examples (setup as above):

```
python3
```

load tokenizer and model (in Python shell):

```
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('LumiOpen/Poro-34B')
model = AutoModelForCausalLM.from_pretrained('LumiOpen/Poro-34B', device_map='auto')
```

confirm that you're on GPU (should show `type='cuda'`)

```
model.device
```

model loss for text:

```
encoded = tokenizer('This is just a short sentence', return_tensors='pt').to(model.device)
output = model(**encoded, labels=encoded.input_ids)
output.loss
```

generation with default parameters:

```
encoded = tokenizer('Hi, my name is', return_tensors='pt').to(model.device)
output = model.generate(**encoded)
tokenizer.decode(output[0])
```

## Quickstart for puhti

Start an interactive session on a GPU node

```
./slurm/gpu-sinteractive-puhti.sh
```

Set up environment

```
module load pytorch
pip install accelerate --user
export HF_HOME=/scratch/project_2007628/hf_cache
```

Test generation with a larger model

```
echo "The best life advice I've ever heard is this:" \
    | python3 generate.py tiiuae/falcon-40b --dtype fp16 --trust-remote-code
```

## Quickstart for mahti

Start an interactive session on a GPU node

```
./slurm/gpu-sinteractive-mahti.sh
```

Set up environment

```
module load pytorch
pip install accelerate --user
export HF_HOME=/scratch/project_2007628/hf_cache
```

Test generation with a larger model

```
    echo "The best life advice I've ever heard is this:" \
        | python3 generate.py tiiuae/falcon-40b --dtype bf16 --trust-remote-code
```
