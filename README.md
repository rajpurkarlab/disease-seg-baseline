# disease-seg-baseline

## Instructions

### Evaluating Disease Segmentation

After running `pip3 install -r requirements.txt`, run

```
python3 run_seg.py [model] [test_set] [visualize] [method]
```

where `model` is the name of the model to use (e.g, `BioViL`), `test_set` is the name of the test set to be used for evaluation (e.g, `CheXlocalize`), `visualize` denotes whether or not to output plots of imags and masks (e.g., `yes` or `no`), and `method` is what phrase grounding method to use (`naive`, `grad_cam`, `cocoa`).

Example command:

```
python3 run_seg.py BioViL CheXlocalize no grad_cam
```

### Finding Best Prompts

Run

```
python3 find_prompts.py [model] [validation_set] [corpus_set] [method]
```

where `model` is the name of the model to use (e.g, `BioViL`), `validation_set` is the name of the validation set to be used (e.g., `CheXlocalize`), `corpus_set` is the set of all report phrases to search over (e.g, `MIMIC-CXR` or `MS-CXR`), and `method` is what phrase grounding method to use (`naive`, `grad_cam`, `cocoa`).

Example command:

```
python3 find_prompts.py BioViL CheXlocalize MS-CXR naive
```

<!--
## Results Leaderboard

### TODO (vramesh): Update leaderboard

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th colspan=6>mIoU by Pathology</th>
            <th>Overall mIoU</th>
        </tr>
        <tr>
            <th></th>
            <th>Cardiomegaly</th>
            <th>Edema</th>
            <th>Consolidation</th>
            <th>Atelectasis</th>
            <th>Pneumothorax</th>
            <th>Pleural Effusion</th>
            <th></th>
        </tr>
    </thead>
    <tbody>
    <tr>
            <td>BioViL w/o modifications</td>
            <td>0.0628</td>
            <td>0.0276</td>
            <td>0.0053</td>
        <td>0.0446</td>
        <td>0.0028</td>
        <td>0.0618</td>
        <td>0.0342</td>
        </tr>
        <tr>
            <td>(Associated Prompts)</td>
            <td>Findings suggesting cardiomegaly</td>
            <td>Findings suggesting an edema</td>
            <td>Findings suggesting consolidation</td>
            <td>Findings suggesting atelectasis</td>
            <td>Findings suggesting a pneumothorax</td>
            <td>Findings suggesting pleural effusion</td>
            <td></td>
        </tr>
        <tr>
            <td>BioViL w/ prompts hard-searched over MS-CXR</td>
            <td><strong>0.0785</strong></td>
            <td><strong>0.0519</strong></td>
            <td><strong>0.0623</strong></td>
            <td><strong>0.1533</strong></td>
            <td><strong>0.0064</strong></td>
            <td><strong>0.0724</strong></td>
            <td><strong>0.0708</strong></td>
        </tr>
        <tr>
            <td>(Associated Prompts)</td>
            <td>heart size is enlarged</td>
            <td>interstitial edema is present in the right lower lung</td>
            <td>There is consolidation of bilateral lung bases, left more than right</td>
            <td>probable left pleural effusion with adjacent atelectasis</td>
            <td>there is a left chest tube and small basilar left pneumothorax</td>
            <td>Left mild pleural effusion is unchanged, and low lung volumes persist</td>
            <td></td>
        </tr>
    </tbody>
</table> -->
