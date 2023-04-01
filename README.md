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
            <th colspan=10>mIoU by Pathology</th>
            <th>Overall mIoU</th>
            <th>Overall Avg. DICE</th>
        </tr>
        <tr>
            <th></th>
            <th>Enlarged Cardiomediastinum</th>
            <th>Cardiomegaly</th>
            <th>Lung Lesion</th>
            <th>Airspace Opacity</th>
            <th>Edema</th>
            <th>Consolidation</th>
            <th>Atelectasis</th>
            <th>Pneumothorax</th>
            <th>Pleural Effusion</th>
            <th>Support Devices</th>
            <th></th>
            <th></th>
        </tr>
    </thead>
    <tbody>
    <tr>
            <td>BioViL w/o modifications</td>
            <td>0.1088</td>
            <td>0.0628</td>
            <td>0.0017</td>
            <td>0.0592</td>
            <td>0.0276</td>
            <td>0.0053</td>
        <td>0.0446</td>
        <td>0.0028</td>
        <td>0.0618</td>
        <td>0.0461</td>
        <td>0.0423</td>
        <td>0.0629</td>
        </tr>
        <tr>
            <td>(Associated Prompts)</td>
            <td>Findings suggesting enlarged cardiomediastinum</td>
            <td>Findings suggesting cardiomegaly</td>
            <td>Findings suggesting lung lesions</td>
            <td>Findings suggesting airspace opacities</td>
            <td>Findings suggesting an edema</td>
            <td>Findings suggesting consolidation</td>
            <td>Findings suggesting atelectasis</td>
            <td>Findings suggesting a pneumothorax</td>
            <td>Findings suggesting pleural effusion</td>
            <td>Findings suggesting support devices</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>BioViL w/ prompts hard-searched over MS-CXR</td>
            <td>N/A</td>
            <td><strong>0.0785</strong></td>
            <td>N/A</td>
            <td>N/A</td>
            <td><strong>0.0519</strong></td>
            <td><strong>0.0623</strong></td>
            <td><strong>0.1533</strong></td>
            <td><strong>0.0064</strong></td>
            <td><strong>0.0724</strong></td>
            <td>N/A</td>
            <td>---</td>
            <td>---</td>
        </tr>
        <tr>
            <td>(Associated Prompts)</td>
            <td>N/A</td>
            <td>heart size is enlarged</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>interstitial edema is present in the right lower lung</td>
            <td>There is consolidation of bilateral lung bases, left more than right</td>
            <td>probable left pleural effusion with adjacent atelectasis</td>
            <td>there is a left chest tube and small basilar left pneumothorax</td>
            <td>Left mild pleural effusion is unchanged, and low lung volumes persist</td>
            <td>N/A</td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table>
