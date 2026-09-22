# U-Net Ensemble Segmentation with Genetic Algorithm Optimisation

Skin lesion segmentation on **ISIC 2018** using five U-Net-based architectures combined
through a **genetic algorithm-optimised weighted ensemble**.

The GA ensemble reaches **0.931 accuracy, 0.878 Dice, 0.804 Jaccard** on the ISIC 2018
test set — beating every individual model and a standard average ensemble, trained
entirely on a single 8 GB laptop GPU.

> MSc thesis, Cyprus International University (2024). Accepted for oral presentation at
> **ICAFS 2025** (17th International Conference on Theory and Application of Fuzzy
> Systems, Iași, Romania) — indexed by Scopus and Web of Science.

📖 **[Full technical documentation →](docs/DOCUMENTATION.md)**

---

## Results

ISIC 2018 Task 1 test set, 1,000 images, 256 × 256 input.

| Model | Accuracy | Dice | Jaccard | Sensitivity | Precision | Thresh. Jaccard |
|---|---|---|---|---|---|---|
| U-Net (baseline) | 0.896 | 0.811 | 0.716 | 0.868 | 0.829 | 0.611 |
| U-Net + MobileNetV3Large | 0.905 | 0.825 | 0.739 | 0.837 | **0.884** | 0.654 |
| U-Net + ConvNeXtBase | 0.895 | 0.815 | 0.724 | 0.858 | 0.844 | 0.622 |
| U-Net + ResNet50V2 | 0.927 | 0.870 | 0.786 | **0.914** | 0.859 | 0.717 |
| U-Net + VGG19 | 0.906 | 0.824 | 0.730 | 0.860 | 0.856 | 0.630 |
| Average ensemble | 0.918 | 0.852 | 0.770 | 0.879 | 0.880 | 0.697 |
| **GA-weighted ensemble** (τ = 0.6) | **0.931** | **0.878** | **0.804** | 0.891 | **0.901** | **0.743** |

*Thresholded Jaccard follows the ISIC 2018 rule: per-image scores below 0.65 count as 0.*

The top five of 200 teams on the ISIC 2018 live leaderboard scored 0.804–0.836
thresholded Jaccard, using purpose-built multi-scale architectures. This approach reaches
0.743 with off-the-shelf backbones, no external data and no custom layers.

![Prediction samples](results/figures/prediction_samples.png)

Median IoU across the full test set is 0.80; 500 of 1,000 images exceed 0.80 and 93
exceed 0.90. The consistent failure mode is low-contrast lesions on pale skin.

---

## How it works

1. **Five segmentation models** — a from-scratch U-Net plus four variants whose encoder
   is an ImageNet-pretrained backbone (MobileNetV3Large, ConvNeXtBase, ResNet50V2,
   VGG19) with a decoder trained from scratch.
2. **Soft prediction export** — each model emits per-pixel probability maps, not
   binarised masks, so confidence survives into the blend.
3. **Genetic algorithm** — searches the five blend weights that maximise Jaccard on the
   **validation** set. Elitism (top 5%), single-point crossover (0.8), Gaussian mutation
   (0.05, annealed), weights normalised to sum to 1. Fitted on validation, reported on
   test.
4. **Threshold sweep** — 0.4 / 0.5 / 0.6; 0.6 gives the best precision/recall balance.

Three selection strategies were compared — elitism (0.8164 best validation fitness),
tournament (0.8128) and microbial (0.8031). Details and the reasoning behind every
parameter are in the [documentation](docs/DOCUMENTATION.md#6-ensembling).

---

## Repository layout

```
docs/DOCUMENTATION.md     full technical reference
src/
├── common/               shared: metrics, data pipeline, inference IO, soft predictions
├── models/               unet, mobilenetv3large, convnextbase, resnet50v2, vgg19
└── ensembling/
    ├── average/          unweighted mean baseline
    └── genetic/          three GA variants
results/
├── figures/              training curves, prediction samples, GA fitness
├── metrics/              per-model test metrics
└── ga_artifacts/         best weights + per-generation fitness (.npy)
```

---

## Setup

```bash
git clone https://github.com/farshid92/UNet-Ensemble-Segmentation.git
cd UNet-Ensemble-Segmentation
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download ISIC 2018 Task 1 from the
[ISIC Challenge archive](https://challenge.isic-archive.com/data/#2018), then point the
code at it:

```bash
export ISIC_DATASET_PATH=/path/to/ISIC_Challenge_Dataset
export TRAINED_MODELS_DIR=/path/to/trained_models
```

Train and evaluate a model:

```bash
cd src/models/resnet50v2
python train.py
python eval.py
```

Run the ensembles once all five models are trained and their soft predictions exported:

```bash
cd src/ensembling/average  && python average_ensemble.py
cd ../genetic              && python ga_ensemble_simple.py
```

The expected dataset layout, every environment variable and the full training
configuration are documented in
[docs/DOCUMENTATION.md §11](docs/DOCUMENTATION.md#11-environment-and-reproduction).

---

## Citation

```bibtex
@inproceedings{cheraghchian2025ensembling,
  title     = {Ensembling U-Net-Based Models for Lesion Segmentation
               Through Genetic Algorithm},
  author    = {Cheraghchian, Farshid and {\"O}zbilge, Emre},
  booktitle = {17th International Conference on Theory and Application
               of Fuzzy Systems (ICAFS 2025)},
  year      = {2025},
  address   = {Ia\c{s}i, Romania}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Author

**Farshid Cheraghchian** — MSc Computer Engineering, Cyprus International University
[GitHub](https://github.com/farshid92) · [LinkedIn](https://linkedin.com/in/farshidcheraghchian)
