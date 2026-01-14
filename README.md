This repository contains the source code used in the study “Measuring Intra-Model Performance Degradation in Used Smartphones.” 

The code supports reproducible analysis of perceptual media quality and system performance across used smartphones of the same model lineage.

**PEAQ**
**Purpose:** PEAQ (Perceptual Evaluation of Audio Quality) estimates perceived audio degradation by comparing a degraded signal against a reference using a psychoacoustic model.

**Usage in this study:** PEAQ is used to evaluate audio playback–capture quality across devices of the same smartphone model lineage. Scores are computed for paired reference–recorded signals under controlled distances, and intra-model differences are analyzed using average difference, win count, and paired statistical testing.

**Notes:** PEAQ is a full-reference metric and requires time-aligned reference audio. Higher scores indicate better perceptual audio quality.
Installation steps and scripts to execute are uploaded in https://github.com/Priyankapopuri08/SmartphoneQualityAnalysis/tree/main/PEAQ

**OPVQ**

**Purpose:** OPVQ is a perceptual video quality metric implemented in the OpenVQ toolkit, designed to model human sensitivity to spatial and temporal video distortions.

**Usage in this study:** OPVQ is used alongside VMAF to assess perceptual video degradation in screen recordings. Agreement between OPVQ and VMAF is analyzed to ensure robustness of observed intra-model quality trends.

**Notes:** OPVQ is a full-reference metric. Using multiple perceptual metrics reduces the risk of conclusions driven by a single quality model.

Installation steps and scripts to execute are uploaded in https://github.com/Priyankapopuri08/SmartphoneQualityAnalysis/tree/main/OPVQ

**VMAF**
**Purpose:** VMAF (Video Multimethod Assessment Fusion) predicts perceptual video quality by combining multiple elementary quality features through machine learning.

**Usage in this study:** VMAF is applied to screen-recorded videos captured on used smartphones and compared against reference videos. Pairwise intra-model comparisons are performed using average score differences, win counts, and statistical significance testing.

**Notes:** VMAF is a full-reference metric and requires spatial and temporal alignment between reference and degraded videos. Scores range from 0 to 100, with higher values indicating better quality.
Installation steps and scripts to execute are uploaded in https://github.com/Priyankapopuri08/SmartphoneQualityAnalysis/tree/main/VMAF

**PESQ**

**Purpose:** PESQ (Perceptual Evaluation of Speech Quality) estimates perceived speech quality by comparing degraded speech signals with clean references using a perceptual auditory model.

**Usage in this study:** PESQ is applied to recorded podcast segments played and captured across devices. Intra-model comparisons are conducted using average differences, win counts, and paired statistical tests.

**Notes:** PESQ is standardized by ITU-T (P.862) and is widely used for evaluating telephony and speech communication quality. Higher scores indicate better perceived speech quality.
Installation steps and scripts to execute are uploaded in https://github.com/Priyankapopuri08/SmartphoneQualityAnalysis/tree/main/PESQ

**Citation**

@inproceedings{usedsmartphones,
  title={Measuring Intra-Model Performance Degradation in Used Smartphones},
  author={Anonymous},
  year={2026}
}



