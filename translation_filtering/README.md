# Translation filtering task

### Structure

1. **generate_translations.py** // Translate English entries from Helsinki-NLP/europarl -dataset using Poro, take the 
resulting (presumed to be poor) translation and append it to the dataset.
    - This one should be working now (haven't tested with the model)
2. **TODO** // Fine-tune Poro to fix poor translations by feeding it the Poro-translation as input
and europarl translation as output

