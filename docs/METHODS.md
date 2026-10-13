# Methods notes

These notes explain the features used in this project and why they were chosen. They describe the approach, not clinical findings.

## Why hand-designed features?
A CNN learns its own representation from pixels. Here the representation is designed up front from two ideas: the *shape* of intensity landscapes (topology) and the *roughness* of structures (fractal geometry). The hypothesis is that pneumonia changes how opacities and texture are organised in the lung fields, and that these global descriptors can capture it with far fewer parameters than a CNN. The hypothesis is tested by the results in the README, not assumed.

## Persistent homology on images
Treat the grayscale image as a height map. Sweep a threshold from 0 to 1 and look at the set of pixels at or below it (a *sublevel set*).

- **Dimension 0 (H0):** connected components. A component is born when the threshold reaches a local minimum and dies when it merges with an older component.
- **Dimension 1 (H1):** loops. A loop is born when a ring of pixels closes around brighter pixels and dies when the enclosed area fills in.

Each feature is recorded as a (birth, death) pair. Long-lived pairs are large structures, and short-lived pairs are mostly noise. The project also runs the same sweep on the inverted image, which captures *superlevel* sets (bright structures).

From each set of pairs, the code computes:
- count, total persistence, mean, standard deviation and maximum of the lifetimes (death minus birth)
- persistence entropy, which measures how evenly the total lifetime is spread
- a 16-point Betti curve: how many features are alive at each threshold

Features that never die (death = infinity) are clipped to the maximum pixel value of 1.0.

## Fractal features
- **Box-counting dimension:** cover a binary image with boxes of size s and count the boxes that contain foreground, N(s). The slope of log N(s) against log(1/s) estimates dimension: about 1 for a thin line, about 2 for a filled region. It is computed on three binary images: Otsu threshold, Canny edges and mean threshold.
- **Lacunarity:** 1 + variance(mass) / mean(mass)^2 over non-overlapping boxes. It measures how clumpy or gappy the texture is, at box sizes 8 and 16.

## Evaluation design
- Images are resized to 128x128 and contrast-enhanced with CLAHE.
- Cross-validation is grouped by patient id (parsed from file names) so images from one patient never sit in both a training and a validation fold.
- The official test split is used once, after model selection.
- `src/ablation.py` measures which feature groups carry the signal, and `src/threshold.py` chooses the decision threshold from train-only out-of-fold predictions, never from the test set.

## Known weaknesses
The dataset is pediatric and single-source, the labels are not independently verified here, and the model has not been validated on any other hospital's images.
