# Branch: NIQE (Natural Image Quality Evaluator)

## Description
NIQE is an opinion-unaware and distortion-unaware no-reference image quality metric. It constructs a Multivariate Gaussian (MVG) model of natural pristine images and measures the perceptual deviation of a test image from this natural baseline.

## Distance Formulation (Section 3.4)
$$D(\nu_1, \nu_2, \Sigma_1, \Sigma_2) = \sqrt{(\nu_1 - \nu_2)^T \left( \frac{\Sigma_1 + \Sigma_2}{2} \right)^{-1} (\nu_1 - \nu_2)}$$

Where:
- $\nu_1, \Sigma_1$ are the mean vector and covariance matrix of selected high-variance patches of the test image.
- $\nu_2, \Sigma_2$ are the pristine natural corpus model parameters.

### Perceptual Score Mapping $Q(x)$
$$Q(x) = \beta_1 \left( \frac{1}{2} - \frac{1}{1 + e^{\beta_2(x - \beta_3)}} \right) + \beta_4 x + \beta_5$$

## Execution
```bash
python3 run_niqe.py [path_to_image]
```
Example:
```bash
python3 run_niqe.py ../../sample_images/Vivo_Y67_2018_indoor.jpg
```
