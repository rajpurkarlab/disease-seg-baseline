# disease-seg-baseline

## Instructions

After running `pip3 install -r requirements.txt`, run

```
python3 run_seg.py [model] [test_set] [visualize]
```

where `model` is the name of the model to use (e.g, `biovil`), `test_set` is the name of the test set to be used for evaluation (e.g, `CheXlocalize`), and visualize denotes whether or not to output plots of imags and masks (e.g., `yes` or `no`).

Example command:

```
python3 run_seg.py biovil CheXlocalize no
```

## Results Leaderboard

| Models                   | mIoU                | Avg. DICE           |
| ------------------------ | ------------------- | ------------------- |
| BioViL w/o modifications | 0.04160316690371632 | 0.06266696649111267 |
