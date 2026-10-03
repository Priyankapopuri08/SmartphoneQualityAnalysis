# Branch: IL-NIQE (Integrated Local Natural Image Quality Evaluator)

## Description
IL-NIQE extends the natural image statistic framework by integrating multiple feature domains:
1. Spatial luminance MSCN distributions across scales
2. Color / Chrominance statistics ($C_b, C_r$ channels)
3. Gradient magnitude edge and texture responses

## Metric Formulation (Section 3.4)
Estimates perceptual degradation by computing the statistical distance from natural pristine multi-domain distributions without requiring reference images or training on human opinion scores.

### Perceptual Score Mapping $Q(x)$
$$Q(x) = \beta_1 \left( \frac{1}{2} - \frac{1}{1 + e^{\beta_2(x - \beta_3)}} \right) + \beta_4 x + \beta_5$$

## Execution
```bash
python3 run_il_niqe.py [path_to_image]
```
Example:
```bash
python3 run_il_niqe.py ../../sample_images/low_light_2018.jpg
```
