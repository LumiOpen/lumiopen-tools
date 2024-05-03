# Translation analysis script package

---

### Setting up

Dataset layouts:
 - ELRC-Norden
   1) Download bilingual moses files 
        from https://opus.nlpl.eu/ELRC-www.norden.org/en&fi/v1/ELRC-www.norden.org
   2) Extract to <proj_root>/data/elrc-norden_en-fi/. Ensure no subfolder.
 - Europarl
   - Dataset should be downloaded automatically. If not, extract HF dataset files to 
        <proj_root>/data/europarl_en-fi/.

### Running

main.py is the primary entrypoint, which can be run with `python3 main.py -d [dataset:str]
-b [bands:int] -p [per_band:int] -t [threshold:float] -m [min_len:int]`.

Setting the dataset argument is required, others are optional and have default values set
in the script.

Python 3.12 was used during development.