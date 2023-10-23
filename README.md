# lumiopen-tools

Tools for working with LumiOpen models

## Quickstart for puhti

Start an interactive session on a GPU node

```
./slurm/gpu-sinteractive-puhti.sh
```

Set up environment

```
module load pytorch
pip install accelerate --user
export TRANSFORMERS_CACHE=/scratch/project_2007628/transformers_cache
```

Test generation with a larger model

```
echo "The best life advice I've ever heard is this:" \
    | python3 generate.py tiiuae/falcon-40b --dtype fp16 --trust-remote-code
```
