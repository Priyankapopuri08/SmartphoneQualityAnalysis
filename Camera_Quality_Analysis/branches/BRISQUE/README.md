# Branch: BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)

## Description
BRISQUE quantifies spatial image distortions without a reference image by evaluating Natural Scene Statistics (NSS) features in the spatial luminance domain.

## Methodology & Formulas (Section 3.4)
1. **MSCN Extraction**:
   $$\hat{I}(i, j) = \frac{I(i, j) - \mu(i, j)}{\sigma(i, j) + C}$$
2. **GGD Fitting**:
   Fits shape $\alpha$ and variance $\sigma^2$ using moment matching.
3. **Directional AGGD Products**:
   Horizontal ($H$), vertical ($V$), and two diagonal ($D_1, D_2$) pairwise products fitted to Asymmetric Generalized Gaussian Distributions (AGGD).
4. **Multiscale Analysis**:
   Extracted at scale 1 and scale 2 (downsampled by factor of 2) yielding 36 total NSS features.
5. **Perceptual Score Mapping $Q(x)$**:
   $$Q(x) = \beta_1 \left( \frac{1}{2} - \frac{1}{1 + e^{\beta_2(x - \beta_3)}} \right) + \beta_4 x + \beta_5$$
   Maps raw objective score $x$ to perceptual quality scale $Q(x) \in [1, 5]$.

## Execution
```bash
python3 run_brisque.py [path_to_image]
```
Example:
```bash
python3 run_brisque.py ../../sample_images/Oppo_A37_2018_daylight.jpg
```
