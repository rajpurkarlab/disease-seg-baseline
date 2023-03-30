# disease-seg-baseline

## Instructions

### Evaluating Disease Segmentation

After running `pip3 install -r requirements.txt`, run

```
python3 run_seg.py [model] [test_set] [visualize]
```

where `model` is the name of the model to use (e.g, `BioViL`), `test_set` is the name of the test set to be used for evaluation (e.g, `CheXlocalize`), and visualize denotes whether or not to output plots of imags and masks (e.g., `yes` or `no`).

Example command:

```
python3 run_seg.py BioViL CheXlocalize no
```

### Finding Best Prompts

Run

```
python3 find_prompts.py [model] [validation_set] [corpus_set]
```

where `model` is the name of the model to use (e.g, `BioViL`), `validation_set` is the name of the validation set to be used (e.g., `CheXlocalize`), and corpus set is the set of all report phrases to search over (e.g, `MIMIC-CXR` or `MS-CXR`).

Example command:

```
python3 find_prompts.py BioViL CheXlocalize MS-CXR
```

## Results Leaderboard

| Models                   | mIoU                | Avg. DICE           |
| ------------------------ | ------------------- | ------------------- |
| BioViL w/o modifications | 0.04160316690371632 | 0.06266696649111267 |
