# Pneumonia detection from chest X-rays using topology and fractal features

Classifies chest X-rays as NORMAL or PNEUMONIA from hand-designed features instead of a learned CNN: persistent homology (topological data analysis) plus fractal measures, fed to classical classifiers.

## Method
1. **Preprocess:** grayscale, resize to 128x128, CLAHE contrast enhancement, scale to [0, 1].
2. **Topological features** (`src/topology.py`): cubical persistent homology of the sublevel filtration of the image and of its inverse (superlevel sets), in dimensions 0 (connected components) and 1 (loops). For each: persistence statistics (count, total, mean, std, max, entropy) and a 16-point Betti curve. 88 features.
3. **Fractal features** (`src/fractal.py`): box-counting dimension of the Otsu-thresholded image, the Canny edge map and the mean-thresholded image, plus lacunarity at two box sizes. 5 features.
4. **Classifiers** (`src/train.py`): logistic regression, RBF SVM, random forest, gradient boosting. Model selection by 5-fold cross-validation grouped by patient on the train split. The official test split is evaluated once with the selected model.

## Results
Cross-validation on the train split (ROC-AUC, mean +/- std over 5 patient-grouped folds):

| Model | CV ROC-AUC |
|---|---|
| Logistic regression | 0.963 +/- 0.010 |
| SVM (RBF) | 0.970 +/- 0.005 |
| Random forest | 0.956 +/- 0.009 |
| Gradient boosting | 0.956 +/- 0.007 |

Selected model (SVM, RBF) on the official test split, default 0.5 threshold:

| Metric | Value |
|---|---|
| ROC-AUC | 0.901 |
| Accuracy | 0.766 |
| Recall (pneumonia) | 0.962 |
| Specificity | 0.440 |
| Precision | 0.741 |
| F1 | 0.837 |

Confusion matrix on test: 103 true negatives, 131 false positives, 15 false negatives, 375 true positives.

The model catches almost all pneumonia cases but flags many normal X-rays as pneumonia. Test ROC-AUC (0.90) is lower than cross-validation AUC (0.97), which points to a distribution shift between the train and test splits. No CNN baseline was run, so this project makes no claim about beating CNNs.

## Structure
```
src/config.py     paths and constants
src/dataset.py    folder indexing, patient ids, image loading
src/topology.py   persistent homology features
src/fractal.py    fractal features
src/extract.py    feature extraction for all splits
src/train.py      classical models, CV and test evaluation
results/          tda_fractal_metrics.json
docs/             pneumonia_ui_mockup.html (interface mockup, no model)
```

## Data (not included)
Kaggle chest X-ray pneumonia dataset, laid out as `chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`. Download it and place `chest_xray/` in the project root.

## Run
```bash
pip install -r requirements.txt
python src/extract.py          # writes features/*.npz
python src/train.py            # writes results/tda_fractal_metrics.json
```

## Limitations
- Low specificity (0.44) at the default threshold. The threshold was not tuned.
- The official validation split is tiny, so selection uses patient-grouped cross-validation on train. Patient ids are parsed from file names and are best-effort.
- The dataset is pediatric (young children) from a single source, so results do not transfer to adults or other hospitals.
- Labels come from the dataset authors and may contain errors.
- `docs/pneumonia_ui_mockup.html` is an interface mockup with placeholder output. It does not run a model.
- Research prototype. Not a medical device and not for clinical use.
