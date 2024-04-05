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
    | python3 generate.py LumiOpen/Poro-34B --dtype fp16 
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
