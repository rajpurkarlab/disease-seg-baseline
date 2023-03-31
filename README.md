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

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th colspan=6>mIOU by Pathology</th>
        </tr>
        <tr>
            <th></th>
            <th>Cardiomegaly</th>
            <th>Edema</th>
            <th>Consolidation</th>
            <th>Atelectasis</th>
            <th>Pneumothorax</th>
            <th>Pleural Effusion</th>
        </tr>
    </thead>
    <tbody>
    <tr>
            <td>BioViL w/o modifications</td>
        </tr>
        <tr>
            <td>BioViL w/ hard-search prompts</td>
            <td>0.0785</td>
            <td>0.0519</td>
            <td>0.0623</td>
            <td>0.1533</td>
            <td>0.0064</td>
            <td>0.0724</td>
        </tr>
        <tr>
            <td>(Associated Prompts)</td>
            <td>heart size is enlarged</td>
            <td>interstitial edema is present in the right lower lung</td>
            <td>There is consolidation of bilateral lung bases, left more than right</td>
            <td>probable left pleural effusion with adjacent atelectasis</td>
            <td>there is a left chest tube and small basilar left pneumothorax</td>
            <td>Left mild pleural effusion is unchanged, and low lung volumes persist</td>
        </tr>
    </tbody>
</table>
