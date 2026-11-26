# Analysis plan — Inter-adapter leakage in exemplar-free class-incremental learning

**Version:** 1.0, 24 Nov 2026, reviewed and approved by Edilene and Takashi. From this version on, the text does not change; any change is added as a dated addendum (Section 10).
**Authors:** Edilene Aparecida Veneruchi de Campos, Edson Takashi Matsubara (FACOM-UFMS, Brazil).
**Where it is kept:** versioned in the project's Git repository, together with the code (Section 9). No registration was made on an external platform (such as OSF). This English version is also included as an appendix of the paper's supplementary material. A Portuguese version with the same content exists (`PLANO_DE_ANALISE.md`).

---

## 0. What this document is for and how to read it

This **analysis plan** describes, **before** the test experiments are run, what we expect to find, how we will measure it and how we will decide whether we were right. Once fixed, it can no longer be adjusted to the results. This protects the work from a common criticism: that the authors tried many things and reported only what worked.

**Why "analysis plan" and not "preregistration".** A formal preregistration is deposited on an external platform (such as OSF), which fixes the date in a way third parties can check. We chose not to make such a deposit. The plan is versioned in the project's Git repository, and its precedence over the test runs relies on the repository's tags (Section 9). **Acknowledged limitation:** Git dates are recorded by the authors themselves and can, in principle, be altered; therefore, we do not claim that precedence can be verified independently. What we offer is transparency: the full plan, the repository history and the decision rules, which make any deviation visible.

The document is written for readers who **did not follow** the development. Section 0.1 explains the technical terms and the names of the arms; Section 7 explains every statistical computation with numerical examples.

### 0.1 Glossary

| Term | Meaning |
|---|---|
| **CIL** (*class-incremental learning*) | The model learns new classes in stages (**tasks**). After each task, it must classify an image among **all** classes seen so far, without being told which task the image came from. |
| **Exemplar-free** | No image from earlier tasks is stored. Only summary statistics (means, covariances) may be stored. |
| **Frozen backbone** | A pre-trained network (ResNet-18 or ViT-B/16) that turns each image into a vector of numbers, the **feature** φ. Its weights never change. |
| **φ (feature)** | Vector of 512 numbers (ResNet-18) or 768 numbers (ViT-B/16) that describes the image. |
| **Adapter** | Small trainable network placed after the backbone. In our method, each task gets its own adapter, which is frozen when the task ends. |
| **Prototype** | Vector that represents a class: the mean of the features (already passed through the adapter) of that class's training images. |
| **Score** | How much an image "looks like" a class. Here, the cosine of the angle between the image's feature and the class prototype (from −1 to 1). The class with the highest score is the prediction. |
| **Leakage (Λ)** | Fraction of test images classified into a class of **another task**. Example: an image of "apple" (task 1) classified as "tulip" (task 3). |
| **Within-task error (E_WP)** | Fraction of images classified into a wrong class **of the right task**. Example: "apple" classified as "orange", both from task 1. |
| **NCM** (*nearest class mean*) | Training-free classifier: compares the image's feature φ with the mean φ of each class and picks the closest by cosine. Used as the reference. |
| **Score standardization** | Correction applied after training: from the scores of each task, subtract the mean μ_k and divide by the standard deviation σ_k that this task's adapter had on its own training data. |
| **Learned refusal** | Teaching each adapter, during its own training, to recognize **what does not belong to its task** and to give it low scores. Without it, an adapter only learns to say "this is from my task", never "this is not mine". |
| **Pseudo-negatives** | Synthetic features of old classes, sampled from stored Gaussians, used for learned refusal: the new adapter is trained not to claim them. |
| **Gaussian bank** | For each class seen so far, the mean and a simplified covariance of φ are stored. This allows sampling features similar to that class's without storing images. |
| **Covariance rank** | A class's covariance describes in which directions its features vary and by how much. Storing it in full costs d × d numbers (768 × 768 ≈ 590 thousand with the ViT). With **rank r**, we store only the r directions in which the class varies most (the r principal components) and treat the rest as a small variation, equal in all of them. Rank 32 = the 32 principal directions. |
| **LVQ** (*learning vector quantization*) | Technique that keeps "units" in feature space, moving them towards examples of their class and away from examples of other classes. It was part of the initial method as an extra loss term; it was switched off during development (Section 1.2). |
| **Task oracle** | Hypothetical situation in which someone tells us the task of each test image, and we only need to choose the class within it. Accuracy in this situation measures only within-task quality, with no possible leakage. |
| **Arm** | An experimental configuration (method or variant) compared with the others. The main ones are below; Section 4.3 details each of them. |
| **Baseline arm** | One adapter per task, trained only on the images of its task, without pseudo-negatives. It is the system whose problem we study. |
| **gauss arm** | The baseline plus pseudo-negatives drawn at random from the Gaussian bank, that is, with learned refusal. |
| **fecam arm** | Training-free classifier that, like NCM, uses only feature statistics, but measures the distance to each class taking into account the shape of that class's point cloud (Mahalanobis distance), not only the angle. Published method (FeCAM, NeurIPS 2023). |
| **gauss_ca arm** | Uses the same Gaussian bank in a different way: instead of training adapters, it retrains a single classifier over all classes with features sampled from the bank ("ca" = *classifier alignment*). It tests whether learned refusal is worth more than this simple use of the Gaussians. |
| **adapter_compartilhado arm** (shared adapter) | A single adapter for all tasks, trained further at each new task. It mimics the situation in which standardization did not work in the literature (Xu et al., 2026). |
| **Seed** | Number that fixes all random draws of a run (class order, initialization, batches). Different seeds = independent repetitions of the experiment. |
| **pp** | Percentage points. From 40% to 44% is +4 pp. |
| **Final accuracy** | Accuracy on the test images of all classes, after the last task. |
| **AAC** | Mean of the accuracies measured after each task. |
| **Paired t-test** | Statistical test that checks whether the difference between two arms, measured seed by seed, is consistently different from zero (Section 7.3). |
| **Holm correction** | Adjustment of p-values when several tests are made at once, so that chance does not produce false "findings" (Section 7.5). |
| **Spearman correlation** | Measures whether two quantities grow together, looking only at the order of the values (which one is larger), not at the values themselves. It goes from −1 to +1 (Section 7.6). |
| **One-sided permutation test** | Checks whether a positive correlation could have arisen by chance: it shuffles the data in every possible way and counts how often a correlation as high as the observed one appears. "One-sided" because only the positive direction matters (Section 7.6). |
| **Tukey transformation** | Raising each feature value to a power (here 0.5, the square root) to make its distribution more symmetric, closer to a Gaussian. It only makes sense for non-negative values, such as the ResNet's. Used by FeCAM. |

---

## 1. What was done before this plan

This plan is written **after** a development phase. So that the reader knows exactly what was seen and decided beforehand, that phase is described here in detail.

### 1.1 Setting used during development

All development used **a single setting**:

| Item | Value |
|---|---|
| Dataset | CIFAR-100 (100 classes of objects and animals, 500 training and 100 test images per class) |
| Task split | 5 tasks of 20 classes ("B0 Inc20" protocol), class order drawn by the seed |
| Backbone | ResNet-18 pre-trained on ImageNet, frozen |
| Image size | 64×64 pixels |
| Seeds | 1993 to 1998 |

### 1.2 Timeline of decisions

The decisions below were made **while looking at test results of seeds 1993 to 1996**. For this reason these four seeds are called **development seeds** and do not count as test.

1. **Starting point.** Per-task adapters on the frozen ResNet-18, cosine classification against prototypes, an LVQ term in the loss and early stopping.
2. **Leakage diagnosis.** Splitting the error into leakage and within-task error showed that almost all error was leakage, and that it almost always went towards newer tasks. Compared with NCM, the adapters' leakage was much larger. This defined the problem studied.
3. **Classifier corrections.**
    - **Logit scale s = 16.** Cosines range only from −1 to 1; without multiplying them by a scale, the softmax in the loss cannot become confident and training stalls.
    - **New prototypes initialized at the mean of φ and recomputed every epoch.** Before, the prototypes used in training and in evaluation differed; with the change, training and evaluation use the same targets.
    - **LVQ switched off** (weights α_lvq = 0.4 and λ_lvq = 0.3 set to 0). The LVQ units started at the origin and never moved during training, so the term did not contribute to learning.
4. **Pseudo-negatives.** To attack leakage during training, the Gaussian bank and the refusal loss were created. Three ways of choosing pseudo-negatives were implemented: direct sampling (gauss), selection by the leakage margin (margin) and, later, margin computed on standardized scores (margin_std, code name `margem_padronizada`).
5. **Learned samplers (GFlowNet).** Training a GFlowNet to sample pseudo-negatives was considered. A controlled 2-dimensional study showed that simple importance sampling already reproduced the desired distribution, and that the GFlowNet only matched it when trained on importance-sampling samples. The idea was dropped.
6. **Early stopping, first version (seed 1993).** Stopping when the validation loss of new classes stopped decreasing ended training **before** the adapter learned to refuse old classes.
7. **Early stopping, second version (seeds 1993–1996).** Tracking both losses (new classes and refusal), accuracy became correlated with the number of epochs trained (Pearson correlation 0.85). In other words, early stopping only hurt. **Decision:** fixed training of 100 epochs per task.
8. **Score standardization.** A diagnostic (the scale share, κ) indicated that part of the leakage came from the adapters' scores being on different scales. Applied after training to the checkpoints of seeds 1993–1996 (trained with early stopping), standardization raised the baseline's final accuracy from 24.0% to 40.6%. **Decision:** every evaluation is done with raw and with standardized scores.
9. **margin_std arm.** Created so that margin-based selection uses the same (standardized) scores that decide the class at test time.
10. **Speed-up.** The features φ are computed once and cached on disk. Since there is no data augmentation, this changes no result, only the running time.
11. **Final development code** run on seeds 1993 to 1998, on 18 and 19 Nov 2026. The code and all results of the development phase are marked with the git tag `v0.9-desenvolvimento` in the repository `eavcampos/CIL_e_GFlowNet` (commit `ec84535484056e6d666ad0b2708a4376a06fbbb4`, 23 Nov 2026). This includes the folders `resultados_parada_val_ce/`, `resultados_parada_ambas/` and `resultados_100epocas_v1/`, from the early-stopping runs (items 6 and 7).

### 1.3 Status of each seed and setting

| Item | Status |
|---|---|
| Seeds 1993–1996 | **Development.** Guided decisions 2 to 9. |
| Seeds 1997 and 1998 | Run with the final code and analysed on 19 Nov 2026. **No decision was made after seeing them.** Treated as already-observed test data (Section 7.9). |
| Seeds 1999–2002 | Never run. |
| ViT-B/16; ImageNet-R, ImageNet-A, OmniBenchmark, VTAB; CIFAR-100 with 20 tasks | Never run. |
| fecam, gauss_ca, adapter_compartilhado arms | Not yet implemented. |

### 1.4 Development results that motivate the predictions

CIFAR-100, 5 tasks, ResNet-18, mean over six seeds (1993–1998):

| Arm | Final accuracy, raw scores | Final accuracy, standardized scores | Leakage Λ, raw | Leakage Λ, standardized | Within-task error, standardized |
|---|---|---|---|---|---|
| NCM | 41.3% | — (not applicable) | 46.7% | — | 12.0% |
| Baseline | 24.6% | 39.4% | 68.5% | 53.6% | 7.0% |
| Gauss | 34.3% | 43.4% | 58.0% | 48.5% | 8.1% |
| Margin_std | 35.3% | 44.1% | 56.8% | 47.4% | 8.5% |

Reminder: for every image, correct + leakage + within-task error = 100%. For example, standardized Gauss: 43.4% + 48.5% + 8.1% = 100%.

**What each arm gains or loses.** All differences below are paired means over the six seeds; "6/6" means the difference had the same sign in every seed.

- **NCM (reference, no training):** 41.3%. It misclassifies 58.7% of the images: 46.7 pp through leakage and 12.0 pp within the task. This leakage is the "normal" one, caused only by similar classes in different tasks.
- **Baseline with raw scores: −16.7 pp relative to NCM** (24.6% against 41.3%; 6/6). The adapters are **better** than NCM within each task (error of 7.0% against 12.0%; with a task oracle, 70.9% against 65.2%), but they leak 21.8 pp more, almost always towards newer tasks. The within-task gain is swallowed by leakage.
- **Standardizing the baseline: +14.8 pp** (from 24.6% to 39.4%; 6/6), without retraining anything. Leakage falls from 68.5% to 53.6% and the within-task error does not change, as expected (Section 1.5). Even so, the standardized baseline remains **1.9 pp below NCM**: it still leaks 6.9 pp more than NCM.
- **Pseudo-negatives (gauss) with raw scores: +9.7 pp relative to the raw baseline** (34.3% against 24.6%), but still 7.0 pp below NCM.
- **Pseudo-negatives after standardization: +4.0 pp relative to the standardized baseline** (43.4% against 39.4%; 6/6). Leakage falls by another 5.1 pp; within-task error rises by 1.2 pp, the price of the adapter devoting part of its capacity to refusing old classes.
- **Combination (standardized gauss): +2.1 pp relative to NCM** (6/6). It is the only configuration, together with the margin ones, that surpasses NCM: it leaks almost as much as NCM (48.5% against 46.7%) and errs less within the task (8.1% against 12.0%).
- **Margin selection (margin_std) relative to random sampling (gauss): +0.7 pp**, a **non-significant** difference (95% interval from −0.2 to +1.6; Section 7.3). Against NCM, +2.8 pp (6/6).

Neither mechanism alone surpasses NCM; together, they do. These results come from **a single setting** and from seeds that partly guided development. That is why they need to be tested.

### 1.5 Why the adapters score on different scales, and how this was noticed

**What "different scales" means.** Each adapter scores (by cosine) the classes of its task. If the adapter of task 4 tends to give scores around 0.8 to any image and the adapter of task 0 around 0.5, the adapter of task 4 wins almost always, even on images of task 0. The comparison across tasks becomes unfair, like comparing marks of two exams graded with different strictness.

**Why it happens.** Three reasons, which add up:
1. **Nothing fixes the level of the scores.** Each adapter is trained with cross-entropy only among the classes of its own task. This loss depends only on **how much the right class beats the other classes of the same task**, not on the absolute value of the scores. Adding the same amount to all scores of a task does not change the loss. Hence the level at which each adapter's scores end up is arbitrary, and nothing forces the adapters to end at the same level.
2. **Tasks differ in difficulty.** In a task with well-separated classes, the adapter easily learns to give high scores; in one with similar classes, the typical scores stay lower.
3. **No adapter learns to refuse.** The adapter is trained to pull the images of its task close to its prototypes. Since it never sees images of other tasks, it also pulls those images close, with no penalty. How much this happens varies from adapter to adapter.

The observed **direction** (newer adapters win) was measured, but its exact cause was **not isolated** during development. It is one of the things that P1 and the recording of each task's statistics μ_k (Section 6) will clarify.

**How the phenomenon was noticed.** Through three pieces of evidence, in this order:
1. **The leakage matrix** showed that the images that left their task went almost entirely to the most recent tasks: 76% of the images of the first task were assigned to the last one, and the final per-task accuracy went from 0.1% (task 0) to 62.7% (task 4). With NCM, which has no adapters, errors spread evenly across tasks.
2. **The standardization test.** Standardization subtracts a constant and divides by another **within each task**. Therefore it cannot change which class wins within a task; it can only change which task wins. If the problem were only genuine overlap, it would not help. Applied to the **same** trained model, it gave +14.8 pp, and the direction index fell from 31.2 to 1.7 (1 = no preference). This is only possible if a large part of the leakage comes from differences in level and spread between the scores of the tasks.
3. **The scale share κ = 0.22:** standardization removed about one fifth of the baseline's leakage, precisely the part in excess of NCM, which is the easiest to recover.

**What has not yet been measured directly:** the values of μ_k and σ_k of each task were not recorded in the development runs. In the test runs they are recorded (Section 6), which allows seeing the scale differences directly and computing the scale dispersion DE.

---

## 2. Objectives

### 2.1 General objective

To determine **when** a CIL system with per-task adapters on a frozen backbone needs only **score calibration** (standardization, applied after training) and when it also needs **each adapter to learn, already during training, to recognize what does not belong to its task** (learned refusal, with pseudo-negatives), using indicators that can be measured **before** applying any correction.

In simple terms: an adapter that wins where it should not may simply be "speaking louder" than the others (a scale problem, fixable afterwards, almost for free), or it may have genuinely learned to claim what is not its own (an overlap problem, which only different training solves). We want to tell the two cases apart and know when each remedy is needed.

### 2.2 Why this matters

Recent CIL methods with pre-trained models (EASE, MOS, TUNA) use one module per task and spend considerable effort choosing the right module at test time. Knowing whether the problem is one of scale (fixable after training at almost no cost) or of genuine overlap (requiring a change in training) indicates which type of solution to use. Moreover, the same standardization was tested by Xu et al. (2026) with a shared encoder and brought no gain; our development data show a large gain with independent adapters. A diagnostic that explains this difference is useful to the field.

### 2.3 Specific objectives

| # | Specific objective | Prediction that tests it |
|---|---|---|
| SO1 | Check whether the excess leakage directed towards newer tasks, seen in development, appears in other datasets, protocols and backbones | P1 |
| SO2 | Check whether indicators measured without applying standardization predict how much it will help | P2 |
| SO3 | Check whether pseudo-negatives add a gain after standardization, and whether using them to train adapters is better than using them to realign a single classifier | P3 |
| SO4 | Test why standardization worked on our data and not on those of Xu et al. (2026). Our explanation is that scale differences arise because each task has **its own** adapter, trained in isolation. When all tasks share **a single** adapter, as in Xu et al., the scores of all of them are produced by the same network and should be on similar scales. We test this with the adapter_compartilhado (shared adapter) arm: if the explanation is right, standardization will barely change its accuracy | P4 |
| SO5 | Show where the complete system (standardization + pseudo-negatives) stands relative to two groups: (a) classifiers that train nothing and use only feature statistics (NCM and FeCAM), a comparison we run ourselves, with full statistics; (b) the strongest published methods (RanPAC, EASE, MOS and TUNA), a comparison made initially with the numbers reported in their papers. We do not expect to surpass group (b); the goal is that the reader knows the size of the gap and the cost (memory and time) of each approach | Secondary question S1 |

### 2.4 Out of scope

Proposing a new state-of-the-art method; tuning hyperparameters per dataset; using exemplars; fine-tuning the backbone in our arms.

---

## 3. Predictions

### 3.1 Measures used in the predictions

All are computed per seed and per setting. The computation of each one is detailed in Section 7.2.

- **Setting:** a dataset × protocol × backbone combination (list in Section 4.1).
- **Standardization gain, Δ_std:** final accuracy of the baseline with standardized scores minus final accuracy of the **same** baseline with raw scores. The model is the same; only the decision rule changes.
- **Excess leakage:** Λ of the baseline (raw scores) minus Λ of NCM.
- **Direction index, D:** how much leakage goes to newer tasks divided by how much goes to older tasks. D = 1 means no preference; D = 30 means 30 times more leakage towards newer tasks. Computed with raw scores only.
- **Scale dispersion, DE:** how much the typical scores of the adapters differ from each other, in units of each adapter's typical spread. Computed from training data only.

**Why κ is not used as a predictor.** The scale share κ = (Λ − Λ̃)/Λ uses Λ̃, the leakage **with standardized scores**. In other words, computing κ requires having already applied standardization. Correlating κ with the standardization gain would mean measuring almost the same thing twice and finding a correlation "for free". D and DE do not depend on standardization and are therefore honest predictors. κ is still reported, as a description only.

### 3.2 P1 — Excess leakage directed towards newer tasks

**In plain words:** when each task has its own adapter, trained without ever seeing the others, the newer adapters "steal" images from old tasks, more than a training-free classifier (NCM) would confuse by chance.

**Why we expect it:** each adapter is trained to give high scores to its classes and is never penalized for giving high scores to images of other tasks. In development, the excess was 21.8 pp and D = 31.2.

**How we measure it, in each setting:**
(a) the leakage difference between the baseline (raw scores) and NCM, seed by seed, with a paired t-test (Section 7.4);
(b) the baseline's direction index D, as the median over seeds.

**Confirmed in a setting** if (a) is positive and significant after Holm correction across settings (Section 7.5) **and** the median D is greater than 1.
**Refuted in a setting** if the mean difference in (a) is less than or equal to zero, or if the median D is less than or equal to 1.
**Inconclusive in a setting** otherwise (for example, a positive but non-significant difference).
**Overall result:** confirmed if confirmed in all 8 settings; partially confirmed if in 5 or more; refuted otherwise.

**What each outcome means:** if P1 fails with the ViT, the problem studied is specific to weak features, and the paper says so.

### 3.3 P2 — Scale indicators predict the standardization gain

**In plain words:** in settings where the adapters "speak at different volumes" (high DE) or where leakage is strongly skewed towards newer tasks (high D), standardization helps more.

**Why we expect it:** standardization only corrects scale differences. If the scales are already similar, there is nothing for it to correct.

**How we measure it:** each of the 8 settings becomes **one point**. For each point we compute the mean Δ_std over the six seeds, the median D and the mean DE. Then we compute the Spearman correlation (Section 7.6) between Δ_std and D, and between Δ_std and DE.

**Confirmed** if at least one of the two correlations is positive with p < 0.05 in the one-sided permutation test, after Holm correction for 2 tests.
**Refuted** if both correlations are less than or equal to zero.
**Inconclusive** otherwise.

**What each outcome means:** if P2 is confirmed, the diagnostic becomes a tool: one can know, before correcting, whether standardization is worth applying. If refuted, the paper reports the effect of standardization without claiming it is predictable.

### 3.4 P3 — Refusal adds a gain after standardization and beats classifier realignment

**In plain words:** (a) even after standardization, teaching the new adapter to refuse synthetic features of old classes improves accuracy; (b) using these features to train adapters is better than using them in the simplest way, which is to retrain a single classifier over all classes (gauss_ca arm).

**Why we expect it:** standardization corrects the scale but does not change **where** the adapter places the images. If a new adapter puts old images close to its prototypes, only training fixes it. In development, (a) gave +4.0 pp; (b) was never tested.

**How we measure it, in each setting:**
(a) final accuracy of gauss (standardized) minus baseline (standardized), seed by seed, paired t-test;
(b) final accuracy of gauss (standardized) minus gauss_ca, seed by seed, paired t-test.
Holm correction is applied across the 8 settings, separately for (a) and for (b).

**Confirmed in a setting** if (a) and (b) are positive and significant. **Overall result** with the same rule as P1.
**Outcome registered in advance:** if (a) holds and (b) fails, the conclusion will be "pseudo-negatives help, but there is no advantage in using them to train adapters rather than to realign the classifier". In that case, learned refusal is removed from the paper's list of contributions.

### 3.5 P4 — With a shared encoder, scale matters little

**In plain words:** if all tasks use **the same** adapter, updated at each task, the scores of all tasks end up on similar scales, and standardization barely changes anything. This would explain why Xu et al. (2026) saw no gain.

**How we measure it, in the 4 settings where the arm runs:**
(a) Δ_std of the shared adapter minus Δ_std of the baseline, seed by seed, paired t-test; a negative difference is expected;
(b) the mean Δ_std of the shared adapter.

**Confirmed** if, in all 4 settings, (a) is negative and significant after Holm **and** the absolute value of (b) is smaller than 1 pp.
**Refuted** if, in any setting, (a) is greater than or equal to zero.
**Inconclusive** otherwise.

### 3.6 Secondary questions (no directional prediction)

- **S1.** How does the combination standardization + pseudo-negatives (gauss arm, standardized scores) compare with training-free methods and with published methods? No directional prediction. It has two parts:
    - **S1a — against NCM and FeCAM**, which we run ourselves: paired comparison with the statistics of Section 7, in all settings.
    - **S1b — against RanPAC, EASE, MOS and TUNA**: in the initial phase, we use the published numbers (Section 4.3.7 and Appendix A). Since each paper reports a single number, from a single run with seed 1993, no statistical test is possible: the comparison is **descriptive**, made against our arms run with the same seed 1993 (Section 4.2). If we later run these methods on the test seeds, the comparison follows Section 7, and the paper states where each number comes from.
- **S2.** Does selecting pseudo-negatives by the margin (margin and margin_std arms) change the result relative to direct sampling (gauss)? In development there was no significant difference. We expect this to remain so, but it is not a prediction.

---

## 4. Experimental design

### 4.1 Settings

| # | Dataset | Classes | Protocol | Tasks | Backbone | Input | Used in |
|---|---|---|---|---|---|---|---|
| 1 | CIFAR-100 | 100 | B0 Inc20 | 5 | ResNet-18 | 64×64 | P1–P4 |
| 2 | ImageNet-R | 200 | B0 Inc20 | 10 | ResNet-18 | 64×64 | P1–P4 |
| 3 | CIFAR-100 | 100 | B0 Inc20 | 5 | ViT-B/16 | 224×224 | P1–P4 |
| 4 | CIFAR-100 | 100 | B0 Inc5 | 20 | ViT-B/16 | 224×224 | P1–P3, ablations |
| 5 | ImageNet-R | 200 | B0 Inc20 | 10 | ViT-B/16 | 224×224 | P1–P4, ablations |
| 6 | ImageNet-A | 200 | B0 Inc20 | 10 | ViT-B/16 | 224×224 | P1–P3 |
| 7 | OmniBenchmark | 300 | B0 Inc30 | 10 | ViT-B/16 | 224×224 | P1–P3 |
| 8 | VTAB | 50 | B0 Inc10 | 5 | ViT-B/16 | 224×224 | P1–P3 |

**"B0 IncX"** means: no class learned before the first task (B0) and X new classes per task.

**Why these settings:**
- **Setting 1** is the development one; it checks whether the results replicate with new seeds.
- **Settings 1, 2 and 3** allow comparing backbones on the same dataset and protocol (1 vs. 3) and the same weak backbone on another dataset (1 vs. 2). This creates the variation of scale across settings that P2 needs.
- **Settings 4 to 8** use exactly the protocols of the MOS paper (Sun et al., AAAI 2025); settings 4, 5 and 6 also match those of TUNA (Wang et al., ICCV 2025). This allows comparing our numbers with published ones. For ImageNet-R, the EASE paper uses Inc5; we adopt Inc20, which is the one of MOS, TUNA and the file `exps/ease_inr.json` of LAMDA-PILOT.
- **Settings 3 and 4** differ only in the number of tasks (5 vs. 20), which shows the effect of having more adapters competing.

**Data:** the copies of each dataset distributed in the LAMDA-PILOT README, already split into training and test, which are those used by SimpleCIL, EASE, MOS and TUNA.

**Why ObjectNet was excluded.** The literature on CIL with pre-trained models often includes a benchmark with 200 ObjectNet classes split into training and test. ObjectNet's license, however, states: *"ObjectNet may never be used to tune the parameters of any model."* Training adapters or computing prototypes with these images violates this clause. Therefore ObjectNet is not used in any setting. **Licenses of the datasets used (checked on 22 and 23 Nov 2026).** None of them forbids training models:

| Dataset | License or terms found |
|---|---|
| CIFAR-100 | no formal license; widespread academic use |
| ImageNet-A, ImageNet-R | MIT (authors' repositories) |
| OmniBenchmark | Creative Commons BY-NC-ND 4.0: non-commercial use and no distribution of modified versions. Our use (research, no redistribution of the data) is compatible |
| VTAB, APER's 50-class set | made of 10 classes from each of 5 datasets. They are RESISC45, DTD, Oxford-IIIT Pets, EuroSAT and Flowers-102, in this order (folders 10–19, 20–29, 30–39, 40–49 and 50–59). Folders are numbered, without names; the composition was confirmed on 23 Nov 2026 from the counts of the local copy, which match the PILOT version (1,796 training and 8,619 test images) and the VTAB-1k protocol (about 1,000 training images per source dataset): 40 test images per class = DTD's official split; ~100 training images per class and 5,400 test images = EuroSAT; ~9 training images per class = Flowers-102. EuroSAT: MIT. Oxford-IIIT Pets: Creative Commons (BY 4.0 in a Zenodo copy; check the official page). DTD, Flowers-102 and RESISC45: no explicit license in the copies consulted. VTAB itself was created to train and evaluate models on these datasets |

Two precautions: (i) the APER author, who built the OmniBenchmark and VTAB subsets distributed through PILOT, warns in the README that he does not have the right to redistribute them; therefore **we do not redistribute data** — the code only points to the original links; (ii) the paper cites the license of each dataset.

**Class order:** `numpy.random.seed(seed)` followed by `numpy.random.permutation(C)`, as in LAMDA-PILOT. Our code uses `RandomState(seed).choice(C, C, replace=False)`; on 22 Nov 2026 we verified that both produce exactly the same order for C = 50, 100, 200 and 300. Class indices follow the alphabetical order of the folders, also as in PILOT. **Exception: VTAB.** The benchmark's authors specify that, for VTAB, classes are **not** shuffled (`"shuffle": false`), so that each task corresponds to one of the 5 source datasets (README of the APER repository and file `exps/simplecil/vtab_10_10_vit-b_simplecil.json`). We follow this: in setting 8, the class order is the same in all seeds, and seeds vary only the training randomness (initialization, batches, pseudo-negatives). Since the folders are named 10 to 59 (always two digits), alphabetical order coincides with numerical order, and each task receives exactly one source dataset.

**Two particularities of VTAB, registered in advance:** (i) the test set is very unbalanced across tasks: the EuroSAT task has 5,400 of the 8,619 test images (63%), so final accuracy and leakage are dominated by it; therefore VTAB is also reported with per-task accuracy; (ii) the Flowers-102 task has 7 to 11 training images per class, fewer than the rank 32 of the Gaussian bank: the covariance of these classes is limited to a few directions and relies on shrinkage. This is a limitation of the setting, not a reason to change hyperparameters.

### 4.2 Seeds

- **Test:** 1997, 1998, 1999, 2000, 2001 and 2002, in all settings and arms, including the ablations.
- **Seed 1993, only for the comparison with published numbers (S1b):** the ncm, fecam, gauss_ca, baseline and gauss arms also run with seed 1993 in settings 4 to 8. Published numbers come from a run with this seed, so the comparison uses the **same class order**. These results do not enter any prediction. Seed 1993 was a development seed only in setting 1 (CIFAR-100, ResNet-18); with the ViT and on the other datasets it was never seen.
- **RanPAC, EASE, MOS and TUNA, if run by us later:** the same six test seeds. If GPU time does not allow it, 1997, 1998 and 1999, and this is stated in the paper.

### 4.3 Arms

Each trained arm is evaluated **twice**, with raw and with standardized scores, without retraining. The ncm, fecam and gauss_ca arms, which have no per-task adapters, are evaluated only in their original form, because all their classes already share a single scale. The published methods (Section 4.3.7) enter, in the initial phase, through the numbers reported in their papers.

#### 4.3.1 ncm — nearest class mean, no training
- **What it stores:** the mean of φ for each class (512 or 768 numbers per class).
- **How it classifies:** computes the cosine between the image's φ and each class mean; picks the largest.
- **What it trains:** nothing.
- **Why it is here:** it is the reference that tells how much leakage is "normal". Since all classes live in the same space, there are no adapters competing. It is identical to SimpleCIL (Zhou et al., IJCV 2024).
- **Serves:** P1 (reference for the excess), S1 and the protocol check (Section 8).

#### 4.3.2 fecam — Mahalanobis-distance classifier, no training
- **What it stores:** mean and full covariance matrix of φ for each class.
- **How it classifies:** uses the Mahalanobis distance, which measures the distance to each class mean taking into account the shape of that class's point cloud (directions in which the class varies a lot weigh less). Picks the closest class.
- **What it trains:** nothing.
- **Implementation:** follows the official repository (Goswami et al., NeurIPS 2023, `dipamgoswami/FeCAM`, checked on 22 Nov 2026), training-free version. Before inversion, each class covariance receives **shrinkage**: α1 × (mean of the diagonal) is added to the diagonal and α2 × (mean of the off-diagonal) off the diagonal, which avoids a nearly singular matrix when there are few images per class. It is then **normalized** to a correlation matrix. Values, copied from the official scripts and configurations:
    - ViT-B/16, CIFAR-100: α1 = 1, α2 = 0, no Tukey transformation (script `FeCAM_vit_cifar100.py`);
    - ViT-B/16, other datasets: α1 = 10, α2 = 10, no Tukey transformation (script `FeCAM_vit_imagenet-r.py`, the only other ViT script on natural images);
    - ResNet-18: α1 = 1, α2 = 1, with Tukey transformation β = 0.5, which raises each feature value to the power 0.5 to bring the distribution closer to a Gaussian (`exps/FeCAM_cifar100.json`).
- **Why it is here:** it stores information comparable to our Gaussian bank (mean and covariance per class). Without it, a reviewer could say that our advantage comes only from storing covariances.
- **Serves:** S1.

#### 4.3.3 gauss_ca — single classifier realigned with synthetic features
- **What it stores:** the same Gaussian bank as our arms (Section 5.1).
- **How it classifies:** a cosine classifier with one vector per class, on φ, without adapters.
- **What it trains:** after each task, only the classifier, with features sampled from the bank for **all** classes seen so far.
- **Fixed details:** vectors initialized at the class means of φ; after each task, 5 epochs; in each epoch, 256 sampled features per class seen; SGD with learning rate 0.01 and momentum 0.9; batch 48; scale s = 16; ReLU on samples only with ResNet-18.
- **Why it is here:** it is the simplest and best-known way of using synthetic features (the "classifier alignment" stage of SLCA, Zhang et al., ICCV 2023). If it is as good as training adapters to refuse, learned refusal is not justified.
- **Serves:** P3(b) and S1.

#### 4.3.4 baseline — per-task adapters without pseudo-negatives
- **What it stores:** the adapter of each task (frozen at its end) and the prototype of each class.
- **How it classifies:** the image goes through the backbone and through **all** adapters; each adapter scores the classes of its task (cosine to the prototype); the highest score over all classes wins.
- **What it trains:** at each task, only the new adapter, with cross-entropy on the images of the current task.
- **Why it is here:** it is the system whose problem we study.
- **Serves:** P1, P2, P3(a), P4.

#### 4.3.5 gauss — baseline with randomly sampled pseudo-negatives
- **What it stores:** the same as the baseline, plus the Gaussian bank.
- **What it trains:** like the baseline, but each batch also receives 48 synthetic features of old classes, sampled at random from the bank and labelled with their old class. Since old adapters and prototypes are frozen, the only way to get these features right is for the new adapter to give them low scores, that is, to learn to refuse them.
- **Why it is here:** it is our refusal mechanism in its simplest form.
- **Serves:** P3 and S1.

#### 4.3.6 adapter_compartilhado — a single adapter for all tasks
- **What it stores:** a single adapter and the prototypes of all classes.
- **What it trains:** the same adapter keeps being trained at each task, with cross-entropy on the classes of the current task. The prototypes of old classes stay as they were at the end of their task.
- **Standardization:** μ_k and σ_k computed at the end of each task k, with the adapter as it is at that moment.
- **Why it is here:** it mimics the situation of Xu et al. (2026), in which all tasks share the same feature extractor. It is the direct test of our explanation.
- **Serves:** P4. Runs in settings 1, 2, 3 and 5.

#### 4.3.7 Published methods: RanPAC, EASE, MOS and TUNA
- **What they are:**
    - **RanPAC** (McDonnell et al., NeurIPS 2023): projects the backbone features into a much higher-dimensional space with a fixed random matrix and a nonlinearity, and classifies by ridge regression on class prototypes, decorrelated through the Gram matrix. In its standard version it tunes an adapter on the first task only. It does not use per-task adapters, but it is one of the strongest methods in the field and barely trains.
    - **EASE** (Zhou et al., CVPR 2024): trains adapters **inside** the ViT layers, one set per task, and synthesizes prototypes of old classes in the new subspaces.
    - **MOS** (Sun et al., AAAI 2025): also uses per-task adapters in the ViT, merges adapters to reduce forgetting, and chooses each image's adapter through a training-free retrieval mechanism.
    - **TUNA** (Wang et al., ICCV 2025): combines per-task adapters with a universal adapter; at test time it chooses the task adapter by the entropy of its predictions and adds its predictions to those of the universal adapter.
- **How they enter, in the initial phase:** through the **published numbers**, in settings 4 to 8, provided the source uses the same protocol: ViT-B/16-IN21K, the same task split and the class order of seed 1993. Appendix A lists the source (paper and table) of each number. In the paper, each copied number is marked "reported in [reference]". A cell with no source under an identical protocol is left empty ("—"); a number from another protocol is never used.
- **Why copying is legitimate:** the check in Section 8.1 tests whether our pipeline reproduces the published SimpleCIL. If it does, our data, transformations and checkpoint are those of the papers, and the numbers are comparable. The check in Section 8.2 reproduces one of the four methods to confirm this.
- **If we run them later:** official LAMDA-PILOT code, unchanged. Hyperparameters: those of the PILOT configuration files where they exist (there are files for CIFAR-100 and ImageNet-R); for the other datasets, those reported in the original papers or their supplementary material; where nothing is reported, those of ImageNet-R. The final choice per dataset is recorded in an addendum before running.
- **Why they are here:** they are the state of the art in exemplar-free CIL with pre-trained models. EASE, MOS and TUNA use per-task adapters, as we do, and address the same problem of choosing the right module; RanPAC is the strong reference that barely trains. Without them, the work cannot be positioned.
- **Serves:** S1b. ViT only. Setting 3 (CIFAR-100 with 5 tasks, ViT) has no published numbers.

#### 4.3.8 margin and margin_std — selecting pseudo-negatives by the margin
Described in Section 4.4 (ablation A4). Run in settings 4 and 5.

### 4.4 Ablations

An **ablation** changes a single piece of the method and observes the effect. The ablations below are **exploratory**: they have no prediction and do not decide any main conclusion, but they help understand why the method works. All run in settings 4 and 5 (ViT, CIFAR-100 with 20 tasks and ImageNet-R), with the six test seeds.

**A1. Where the standardization statistics come from.**
- **Question:** would standardization work better if μ_k and σ_k were computed on images the adapter did not see during training?
- **Why ask:** after 100 epochs, the adapter gives very confident scores to its training images. Statistics computed on them may not represent the scores of new images.
- **How:** the baseline is trained on 90% of the training images of each class; the remaining 10% are held out. The **same** model is evaluated in two ways: with μ_k, σ_k computed on the 90% used for training, and with μ_k, σ_k computed on the 10% not seen.
- **Values:** {training statistics, validation statistics}.
- **Reading:** if the validation version is clearly better, part of the leakage left after standardization is poor calibration, not genuine overlap.

**A2. Weight of the refusal loss, λ_pn.**
- **Question:** how much weight should refusal have relative to the classification of the new classes?
- **How:** gauss arm with λ_pn in {0.25; 0.5; 1.0}. The main value is 0.5.
- **Reading:** a trade-off is expected: more weight reduces leakage but increases within-task error. If the result is stable across the three values, the method does not depend on fine-tuning this parameter.

**A3. Rank of the Gaussian bank.**
- **Question:** how much detail of each class's covariance must be stored?
- **How:** gauss arm with covariance rank in {8; 32}. The main value is 32. Rank 8 stores about a quarter of the memory.
- **Reading:** if rank 8 gives a similar result, the method can store much less memory per class.

**A4. How to choose the pseudo-negatives.**
- **Question:** is it worth choosing the pseudo-negatives that the new adapter is "stealing", instead of sampling them at random?
- **The leakage margin.** For a synthetic feature z of the old class c, the margin is

  m(z, c) = (highest score that some **new** class gives to z) − (score that the **true** class c gives to z).

  If m > 0, some new class beats the true class: z **would leak**. The larger m, the more the new adapter is "stealing" z. If m < 0, the true class wins, and the new adapter already refuses z.
- **How the choice is made, at every training batch:**
    1. Draw M = 4096 **candidates** from the Gaussian bank (old classes chosen at random, and one feature sampled from each one's Gaussian).
    2. For each candidate i, compute the margin m_i with the new adapter **in its current state** of training.
    3. Each candidate receives the weight w_i = exp(β × m_i), and the weights are normalized to sum to 1: p_i = w_i / (w_1 + … + w_M).
    4. Draw K = 48 candidates, with replacement, each with probability p_i. These 48 are the batch's pseudo-negatives.

  This procedure is called sampling-importance-resampling: candidates come from a distribution that is easy to sample (the bank), and the weights shift the choice towards the distribution we want (that of the leaking ones).
- **The role of β.** β controls **how strongly** the choice favours the candidates that leak:
    - **β = 0:** all weights equal exp(0) = 1, all candidates have the same chance, and the result equals direct sampling. This is the gauss arm.
    - **Large β:** the weights grow very fast with the margin, and almost only the candidates that leak most are chosen.
    - **Example:** two candidates, one with margin +0.2 (leaks) and one with −0.2 (does not leak). The ratio between their weights is exp(β × 0.4). With β = 5, it equals exp(2) ≈ 7.4: the leaking one is 7.4 times more likely to be chosen. With β = 0.5, it equals exp(0.2) ≈ 1.2: almost the same chance.
    - **Why β = 0.5 in margin_std and β = 5 in margin:** raw scores are cosines, and margins between them are of the order of tenths; standardized scores are measured in standard deviations, and margins between them are of the order of units, about ten times larger. Using a β ten times smaller on standardized scores keeps a degree of preference of the same order in both arms.
    - **Beware of β too large:** if a few candidates concentrate almost all the weight, the 48 chosen are almost always the same, and training sees little variety. We measure this by the **effective sample size**, (Σ w_i)² / Σ w_i², divided by M. In development, it had a median of 17% and a minimum of 8%, far from this problem.
- **The three forms compared, with everything else equal:**
    - **gauss** (β = 0): direct sampling from the bank;
    - **margin** (β = 5): margin computed with raw scores;
    - **margin_std** (β = 0.5): margin computed with standardized scores, the same used to decide the class at test time. Since the task being trained has not ended yet, its μ and σ are estimated during training by a running average over the batches.
- **Reading:** answers secondary question S2. In development there was no significant difference.

**Built-in comparison, no extra run:** settings 3 and 4 show the effect of the number of tasks (5 vs. 20) with the same backbone and dataset.

### 4.5 Conditional exploratory analysis: the diagnostic applied to the published methods

**Condition:** performed only if EASE, MOS and TUNA are run by us (Section 4.3.7). It cannot be done with published numbers, because it needs the prediction for each image. Being exploratory, it has no prediction and does not enter P1 to P4; it is described here so that, if performed, it does not count as an analysis invented after seeing the data.

**Motivation.** Our adapters sit after the backbone, not inside the ViT blocks as in the three methods, because the refusal mechanism needs a fixed feature space in which the Gaussians of old classes remain valid (Section 4.3.7 and the paper's Method section). A reviewer may ask whether the diagnostic applies to these real methods or only to our simple system. This analysis answers that question.

**How each method chooses the task at test time** (checked in the LAMDA-PILOT code on 23 Nov 2026):
- **EASE:** concatenates the features of all adapters into a single vector and classifies with prototypes over that vector. There is no explicit choice of adapter.
- **MOS:** makes a first prediction with the merged-adapter model, goes to the adapter of the predicted class's task, predicts again and repeats until the task stops changing (at most 4 times). At the end, it adds this prediction to that of the merged model.
- **TUNA:** each task adapter produces a prediction; the adapter whose prediction has the **lowest entropy** (the most "confident") is chosen, and its prediction is added to that of the universal adapter.

**What we measure, in settings 4 to 8, on the seeds where the methods are run:**

1. **Leakage diagnostic, in the three methods:** leakage Λ, within-task error E_WP, leakage matrix and direction index D, computed exactly as in Section 7.2, from the predicted and the true class of each test image. Question: is these methods' leakage also larger than NCM's, and does it also go towards newer tasks?
2. **Adapter-choice accuracy, in MOS and TUNA:** fraction of test images for which the chosen adapter is the one of the true task. It is the most direct measure of leakage at the moment of choice.
3. **Standardizing the choice, in TUNA only:** choosing by lowest entropy assumes that the adapters' entropies are on the same scale, the same assumption that our standardization corrects. For each adapter k, we compute the mean μ_k and the standard deviation σ_k of the entropy of its predictions on the training images of task k, at the end of task k. At test time, we choose the adapter with the lowest standardized entropy, (entropy − μ_k)/σ_k. We compare final accuracy and adapter-choice accuracy with and without this standardization, on the **same** trained model. We also compute the scale dispersion DE of the entropies (Section 7.2, with entropies in place of scores).
4. **In MOS and EASE, standardization is not applied:** MOS chooses the adapter by the predicted class, not by comparing scores across adapters, and EASE does not choose an adapter. There is no comparison of scales to correct without changing the method.

**Change to the methods' code:** logging only. We add to the PILOT code the recording, per test image, of the true class, the predicted class, the chosen adapter (MOS and TUNA) and the per-adapter entropies (TUNA). No line affecting training or prediction is changed, and the added code is released with ours. The standardization of item 3 is computed afterwards, from these records.

**Statistics:** the same as in Section 7 (paired differences per seed, t-test, 95% interval), with the caveat that everything here is exploratory and labelled as such in the paper.

---

## 5. Frozen hyperparameters

### 5.1 Common to all settings (values from the development runs)

| Parameter | Value | What it is |
|---|---|---|
| Adapter | residual, d → 64 → d, up-projection initialized at 0 | The adapter adds a correction to φ; it starts as the identity |
| Classifier | cosine to prototypes; new prototypes initialized at the mean of φ and recomputed every epoch | — |
| Logit scale s | 16 | Multiplies the cosines inside the loss |
| Optimizer | SGD, learning rate 0.01, momentum 0.9, weight decay 0 | — |
| Batch | 48 images | — |
| Epochs per task | 100, no early stopping | — |
| Data augmentation | none | This is why φ can be cached |
| Pseudo-negatives per batch, K | 48 | — |
| Refusal weight, λ_pn | 0.5 | Weight of the second term of the loss |
| Gaussian bank | rank-32 covariance plus an isotropic term; shrinkage 10⁻³ | Stores each class's 32 principal directions of variation |
| margin / margin_std | 4096 candidates; β = 5 / β = 0.5 | — |
| Standardization | μ_k, σ_k of adapter k's highest score on each training image of task k, computed at the end of task k | — |

### 5.2 Changes forced by the architecture

Decided now, before any run with the ViT.

| Parameter | ResNet-18 | ViT-B/16 | Reason |
|---|---|---|---|
| Checkpoint | torchvision, ImageNet-1k | `vit_base_patch16_224_in21k` (timm), ImageNet-21k | The same as SimpleCIL and EASE, to compare with the literature |
| d | 512 | 768 | Feature size of each network |
| Image transformation | CIFAR: Resize(64). Folders: Resize(73) + CenterCrop(64). ImageNet normalization | Resize(256, bicubic) + CenterCrop(224); no normalization, as in PILOT | Comparability |
| ReLU on bank samples | yes | **no** | The ResNet feature is never negative (it comes out of a ReLU), so negative samples would be unrealistic. The ViT's can be negative |

No other hyperparameter changes across backbones or datasets. If training fails with the ViT (loss that does not decrease in the first epochs of the first task), the fix is chosen using only the 10% validation split of a single seed (1997) and documented in an addendum before any other run.

---

## 6. Recorded metrics

In each run, after each task and at the end: accuracy, leakage Λ, within-task error E_WP, leakage matrix L, direction index D, scale dispersion DE, scale share κ, task-oracle accuracy and average forgetting. Also: memory stored per class, training time per task and the values μ_k and σ_k of each task (needed for the scale dispersion DE; they were not recorded in the development runs). The computation of each is in Section 7.2.

---

## 7. Analysis

### 7.1 General idea

Each seed is a **repetition** of the experiment. Within a seed, all arms see exactly the same class order. Therefore comparisons are made **seed by seed** (a **paired** comparison): first we compute the difference between two arms in each seed; then we ask whether these six differences are consistently different from zero.

Example: if in seed 1997 arm A reaches 44% and arm B 42%, the difference is +2 pp. We do this for the six seeds and analyse the six differences. This removes the variation that comes from the class order (some orders are harder for all arms), which makes the comparison much more sensitive.

### 7.2 How each metric is computed

All are computed on the test images of all classes seen so far.

- **Accuracy:** number of correctly classified images divided by the total.
- **Final accuracy:** the accuracy after the last task.
- **AAC:** the mean of the accuracies measured after each task. With 5 tasks, it is the mean of 5 numbers.
- **Leakage Λ:** number of images whose predicted class belongs to another task, divided by the total.
- **Within-task error E_WP:** number of images whose predicted class is wrong but from the right task, divided by the total. Always: 1 − accuracy = Λ + E_WP.
- **Leakage matrix L:** T × T table in which row i, column j, is the fraction of images of task i classified into a class of task j. The diagonal holds the images that stayed in the right task.
- **Direction index D:** the sum of the entries of L above the diagonal (images that went to newer tasks) divided by the sum of the entries below the diagonal (images that went to older tasks).
- **Scale dispersion DE:** for each task k we have μ_k and σ_k (mean and standard deviation of adapter k's highest score on its training images). DE = standard deviation of the μ_k across tasks ÷ mean of the σ_k. Example: if the task means are 0.60, 0.70 and 0.80 (standard deviation 0.10) and the typical spreads are 0.05, DE = 0.10 / 0.05 = 2: the means differ by twice each adapter's typical spread.
- **Scale share κ:** (Λ − Λ̃) / Λ, where Λ̃ is the leakage with standardized scores.
- **Task-oracle accuracy:** accuracy when the decision is restricted to the classes of each image's true task. It measures within-task quality only.
- **Average forgetting:** for each task (except the last), the highest accuracy it had at any point minus its accuracy at the end; then the mean over tasks.
- **Memory per class:** number of values stored per class × 4 bytes (32-bit numbers). Example: the rank-32 bank with d = 768 stores (32 + 1) × 768 + 33 = 25,377 values ≈ 101 KB per class.
- **Time per task:** wall-clock training time of each task, measured by the script itself.

### 7.3 Paired t-test, step by step

Used in P1, P3, P4 and S1a (and in S1b, if the published methods are run by us). Let A and B be two arms and n = 6 seeds.

1. **Difference in each seed:** d_s = result of A − result of B, for s = 1, …, 6.
2. **Mean of the differences:** d̄ = (d_1 + … + d_6) / 6. It is the estimated effect, in pp.
3. **Standard deviation of the differences:** s_d = square root of [ Σ (d_s − d̄)² / (n − 1) ]. It measures how much the differences vary from one seed to another.
4. **Standard error of the mean:** SE = s_d / √n. It measures the uncertainty about d̄.
5. **t statistic:** t = d̄ / SE. It says how many "standard errors" the mean is away from zero.
6. **p-value:** the probability of obtaining a t as far from zero as the observed one **if** the two arms were, in fact, equivalent. Computed from Student's t distribution with n − 1 = 5 degrees of freedom, two-sided (it counts departures in both directions). A small p-value (below 0.05, after the correction of Section 7.5) = difference considered real.
7. **95% confidence interval (95% CI):** CI = d̄ ± 2.571 × SE.
    - **What it expresses.** d̄ is our best estimate of the effect, but with another six seeds a slightly different value would come out. The interval shows the range of values of the **true** effect that are compatible with the data. In technical terms: if the experiment were repeated many times, with six new seeds each time, and the interval computed in the same way, 95% of these intervals would contain the true effect.
    - **Where 2.571 comes from.** With many seeds we would use 1.96, the value of the normal distribution that leaves 2.5% on each side. With only six seeds, the standard deviation s_d is itself estimated with uncertainty, and the correct distribution is Student's t with n − 1 = 5 degrees of freedom, which has wider "tails". The value that leaves 2.5% in each tail of this distribution is 2.571. This is why the interval with six seeds is about 30% wider than it would be with many seeds.
    - **How to compute it, in the clear-effect example below:** d̄ = 14.85 and SE = 0.11. Then 2.571 × 0.11 = 0.28, and CI = 14.85 ± 0.28 = [14.57; 15.12].
    - **How to read its width.** Narrow interval = effect measured precisely. Wide interval = much variation across seeds, imprecise effect.
    - **Relation to the p-value.** If the interval **does not contain zero**, the p-value (before correction) is below 0.05; if it contains zero, it is above. The two pieces of information agree; the interval also tells the plausible size of the effect.
    - **The reported intervals are not Holm-corrected.** The "significant or not" decision uses the corrected p-values (Section 7.5); the intervals show the size and precision of the effect.

**Example with a clear effect** (development: standardized baseline − raw baseline, seeds 1993–1998):
differences = 14.54; 15.25; 15.06; 14.65; 14.74; 14.83 pp → d̄ = 14.85; s_d = 0.27; SE = 0.11; t = 137; p < 0.001; 95% CI = [14.57; 15.12]. All seeds agree, and the effect is huge relative to the variation.

**Example with an uncertain effect** (margin_std − gauss, both standardized):
differences = 2.13; 0.55; 0.17; 0.21; 1.43; −0.23 pp → d̄ = 0.71; s_d = 0.89; SE = 0.36; t = 1.95; p = 0.11; 95% CI = [−0.23; 1.65]. The mean is positive, but the interval contains zero: we cannot claim there is an effect.

**Why the t-test and not the Wilcoxon test:** the Wilcoxon test, which does not assume a normal distribution, cannot give a p-value below 0.031 with 6 seeds. After any correction for multiple comparisons, no result could be significant. The number of seeds with the expected sign (Section 7.7) serves as an assumption-free check.

### 7.4 Application to the predictions

| Prediction | A | B | Metric | Where |
|---|---|---|---|---|
| P1(a) | baseline (raw) | ncm | leakage Λ | each setting |
| P3(a) | gauss (standardized) | baseline (standardized) | final accuracy | each setting |
| P3(b) | gauss (standardized) | gauss_ca | final accuracy | each setting |
| P4(a) | adapter_compartilhado | baseline | Δ_std | settings 1, 2, 3, 5 |
| S1a | gauss (standardized) | ncm and fecam (each) | final accuracy | each setting |
| S1b (only if run by us) | gauss (standardized) | ranpac, ease, mos, tuna (each) | final accuracy | settings 4 to 8 |

### 7.5 Correction for multiple comparisons (Holm)

**The problem:** at the 5% level, each test has a 5% chance of coming out "significant" by pure chance, even with no real effect. With 8 settings, the chance of at least one false positive exceeds 33%. The Holm correction controls this: it ensures that the chance of **any** false positive within a family of tests is at most 5%.

**Families:** each prediction is a family. P1(a), P3(a) and P3(b) have 8 tests each (one per setting); P4(a) has 4; P2 has 2 (one per predictor); S1a has one test per setting and per compared method.

**Procedure, for a family of m tests:**
1. Sort the m p-values from smallest to largest: p(1) ≤ p(2) ≤ … ≤ p(m).
2. Compare p(1) with 0.05/m. If it is smaller, test 1 is significant; go to the next. If not, stop: no test is significant.
3. Compare p(2) with 0.05/(m − 1). If it is smaller, test 2 is significant; go to the next. If not, stop.
4. Continue in this way, with threshold 0.05/(m − k + 1) for the k-th test, until the first one that fails. It and all the following are not significant.

**Example with 3 settings**, p-values 0.010; 0.030; 0.045:
- 0.010 < 0.05/3 = 0.0167 → significant;
- 0.030 < 0.05/2 = 0.025? No → stop.
Only the first setting is significant, although all three have p < 0.05 on their own.

The paper reports adjusted p-values, which are equivalent to the procedure above: the adjusted p-value of the k-th test is the larger of (m − k + 1) × p(k) and the adjusted value of the previous test, capped at 1.

### 7.6 Spearman correlation and permutation test (P2)

**What it is:** the Spearman correlation measures whether two quantities grow together, without requiring a straight-line relationship. It goes from −1 (one rises while the other falls) to +1 (they rise together in the same order).

**How to compute it, step by step:**
1. For each of the 8 settings: mean Δ_std over the six seeds and the predictor (median D, or mean DE).
2. Replace the values by their **ranks**: the smallest Δ_std gets rank 1, the largest rank 8; the same for the predictor. Ties receive the mean of their ranks.
3. Compute the ordinary (Pearson) correlation between the two lists of ranks. The result is Spearman's ρ (rho).

**One-sided permutation test:** we ask how often a correlation as high as the observed one would appear if there were no relationship at all. To do so, we shuffle the pairing between the 8 values of Δ_std and the 8 values of the predictor in **every** possible way (8! = 40,320 shuffles), compute ρ for each, and the p-value is the fraction of shuffles with ρ greater than or equal to the observed one. "One-sided" because the prediction has a direction (positive correlation). With 8 points, the computation is exact, with no approximation.

**Bootstrap confidence interval:** to show the uncertainty of ρ, the following is repeated 10,000 times: in each setting, draw 6 seeds with replacement from the 6 available (some appear twice, others not at all), recompute each setting's means and ρ. The 95% interval goes from the 2.5th to the 97.5th percentile of the 10,000 values of ρ.

**Correction:** Holm with 2 tests (D and DE).

### 7.7 Complementary measures, always reported

- **Seeds with the expected sign:** how many of the six seeds show the difference in the predicted direction (for example, 6/6). It is a simple check that does not depend on statistical assumptions.
- **Effect size:** the mean of the differences in pp, with its 95% interval. An effect can be significant and too small to matter, or large and uncertain; the two numbers together show this.

### 7.8 General rules

- **Everything is reported:** confirmed, refuted and inconclusive predictions, with the same prominence.
- **Analyses not listed here** appear in the paper labelled as exploratory. The ablations (Section 4.4) and the analysis of Section 4.5 are exploratory by definition.
- **Runs that fail with an error** are repeated with the same seed. No completed run is discarded.

### 7.9 Seeds 1997 and 1998 in setting 1

Since they have already been observed, the conclusions for setting 1 are reported twice: with the six test seeds and with only the four never-seen ones (1999–2002). If the two analyses disagree, the four-seed analysis prevails.

### 7.10 Summary of the decision rules

| Prediction | Confirmed when | Refuted when |
|---|---|---|
| P1 | in each setting: excess leakage > 0 and significant (Holm, 8 settings) and median D > 1; overall: all settings | excess ≤ 0 or D ≤ 1 |
| P2 | ρ > 0 with p < 0.05 (permutation, Holm with 2 tests) for D or DE | ρ ≤ 0 for both |
| P3 | in each setting: gauss > baseline and gauss > gauss_ca, both significant (Holm, 8 settings) | either difference ≤ 0 |
| P4 | in the 4 settings: shared-adapter Δ_std < baseline Δ_std, significant (Holm, 4), and |shared-adapter Δ_std| < 1 pp | shared-adapter Δ_std ≥ baseline Δ_std in some setting |

---

## 8. Protocol checks

### 8.1 NCM reproduces the published SimpleCIL (before looking at any prediction result)

NCM's final accuracy does not depend on the task size: after the last task, it decides among all classes in the same way. Therefore, with ViT-B/16-IN21K, it must reproduce the published SimpleCIL:

| Dataset | Published SimpleCIL | Source |
|---|---|---|
| CIFAR-100 | 81.26% | EASE (Table 1); TUNA (Table 1) |
| ImageNet-R | 54.55% | EASE (Table 1); TUNA (Table 1) |
| ImageNet-A | 48.91% | EASE (Table 1); TUNA (Table 1) |
| OmniBenchmark | 73.15% | EASE (Table 1) |
| VTAB | 84.38% | EASE (Table 1) |

A difference larger than 2 pp on any dataset indicates a protocol error (data, transformation or checkpoint). In that case, the error is fixed and documented in an addendum before any analysis of the predictions, and the affected runs are redone. If the fix changes the code, the code receives a new tag.

### 8.2 Reproducing a published method

To show that the copied numbers are comparable with ours, we run **RanPAC** on ImageNet-R (setting 5), with the LAMDA-PILOT code and configuration and seed 1993. It is the cheapest of the four methods and one of the strongest. The published value is 77.94% final accuracy (TUNA, Table 1). If our run is within 2 pp of it, we consider the copying of numbers validated. If not, we investigate the difference, and the paper reports the published numbers with this caveat stated explicitly.

---

## 9. Code

**9.1 Plan and code of the test runs.** This plan (version 1.0) and the code of the test runs are kept in the repository `eavcampos/CIL_e_GFlowNet`. Before the first test run, the code is marked with the git tag `v1.0-teste`, and the tag is pushed to GitHub. The commit of this tag contains this plan unchanged. An addendum (Section 10) gives the commit hash. Test runs use only this code.

**9.2 What can and cannot be checked.** After publication, the full repository becomes public: any reader can see the history, the tags `v0.9-desenvolvimento` and `v1.0-teste`, and compare the plan, the code and the results. During a double-blind review, the reviewer receives the code through an anonymous mirror (for example, Anonymous GitHub), which does not show the history, and this plan as an appendix of the supplementary material. In neither case can the dates be confirmed by a source independent of the authors, because no external deposit was made (Section 0). The paper states this explicitly and describes the procedure as an "analysis plan defined before the test runs and versioned in the repository", without claiming independent verification.

**9.3 Clean version for the paper.** Close to submission, a version of the repository containing only what the paper uses is created and marked with the tag `v1.0-artigo`. It may only remove unused files, rename and document; no line that affects training or evaluation may change. Equivalence is verified by running one seed with `v1.0-artigo` and checking that the results are identical to those of the same seed with `v1.0-teste`. The anonymous mirror for review and the archival DOI (Zenodo), after acceptance, are generated from `v1.0-artigo`.

---

## 10. Addenda

Any deviation from this plan is described here, with date, reason and expected effect, in a dedicated commit made **before** the deviation is applied.

| Date | Deviation | Reason |
|---|---|---|
| 24 Nov 2026 | **Description fix (no change of method).** Section 5.1: weight decay is **10⁻⁴ on the Adapter parameters** (`treino.weight_decay_adapter`) and 0 on the others (`otimizador.weight_decay`), not "0" as written. This is the value used in all development runs and kept in the test runs. | Misreading of the configs when drafting the plan, found while implementing Phase B. |
| 24 Nov 2026 | **Description fix.** Sections 1.5 and 6: μ_k and σ_k of the development runs are not in the JSONs, but **they are stored in the checkpoints** (`estatisticas_score_por_tarefa`). The scale dispersion DE of those runs can therefore be computed from them. In the test runs, μ_k and σ_k are also written to the JSON. | Found while implementing Phase B. |
| 25 Nov 2026 | **Change to the adapter_compartilhado arm (Section 4.3.6), before freezing the code.** In this arm, μ_k and σ_k of all tasks seen so far are recomputed after each task with the **current** Adapter, on 256 samples per class from the Gaussian bank (no stored images). The end-of-task statistics are still recorded, for reference only. The other arms do not change. | In the quick code test (2 epochs, seed 1993, setting 1), standardization dropped this arm's accuracy from 13.8% to 2.4%. Since the shared Adapter keeps changing after each task ends, the statistics computed at its end become stale, and P4 would measure this defect rather than the similarity of scales it is meant to test. Remaining difference from the baseline: there the statistics come from real images; here, from bank samples. |
| 25 Nov 2026 | **Implementation detail of the fecam arm (Section 4.3.2).** We follow the official repository: unit-norm vectors, per-class mean and covariance, shrinkage (α1, α2) and normalization to a correlation matrix, pseudo-inverse and Mahalanobis distance. With ResNet-18, the Tukey transformation is applied **from the first task** (in the official code, it only starts at the second task, the first one using means only). | Simplification to use a single rule for all tasks; affects only the fecam arm in settings 1 and 2. |
| 26 Nov 2026 | **Candidate arm, exploratory unless it passes the rule below: Mahalanobis head on the per-task Adapters (before freezing the code).** At the end of task k, using the training images of task k (without the validation part), the mean and covariance of each class are computed in the space of the normalized output of Adapter k, with the same shrinkage, the same correlation normalization and the same α1, α2 as the fecam arm for that backbone and dataset (no Tukey transform, because Adapter outputs can be negative). The class score is minus the Mahalanobis distance; the standardized version (μ_k, σ_k of the winning score on the training images of task k) is also reported. It is evaluation only: it does not change the training of any arm and is recorded in every run with per-task Adapters (`cabeca_mahalanobis` block of the JSON; not applicable to adapter_compartilhado). In the NCM arm, where the Adapter is the identity, the head coincides with FeCAM without Tukey; this serves as a code check. **Decision rule, fixed before any result of the head:** only the development seeds 1993–1996 are used, in scenario 1 (development checkpoints, re-evaluated without training, with `reavaliar_mahalanobis.py`) and in scenario 5 (new full runs of the ncm, baseline, gauss, margem_padronizada and fecam arms). There are six candidate variants: the head on baseline, gauss or margem_padronizada, with raw or standardized scores. The one with the highest mean accuracy, pooling both scenarios, is chosen. It enters the plan as a confirmatory arm (new Section 4.3.9 and prediction P5, "beats FeCAM", tested like P1: one-sided paired t-test per scenario, Holm over 8 tests) only if its mean is **at least 1 pp above FeCAM's in each of the two scenarios** and above FeCAM in at least 3 of the 4 seeds of each scenario. Otherwise it is reported only as exploratory analysis, and the rest of the plan does not change. The test seeds (1997–2002) are not used in this decision. | In the code check (seed 1993, 25 Nov 2026), FeCAM, with no training at all, reached 54.7% in scenario 1, above the best development arm (margem_padronizada, 44.1%, mean of 6 seeds), with lower leakage (36.0% vs about 47%) and higher task-oracle accuracy (75.4% vs about 71%). In scenario 5 it reached 62.2% (NCM: 54.5%). The idea of this head came **after** seeing these numbers; that is why the decision rule is fixed here, before measuring it, and uses only development seeds. |

---

## Appendix A — Source map of the published numbers

Final accuracy (%), ViT-B/16-IN21K, class order of seed 1993, no exemplars. "TUNA T1" = Table 1 of the TUNA paper; "EASE T1" and "MOS T1", likewise. Every cell must be checked against the original PDF before use in the paper.

| Method | CIFAR-100 B0 Inc5 | ImageNet-R B0 Inc20 | ImageNet-A B0 Inc20 | OmniBenchmark B0 Inc30 | VTAB B0 Inc10 |
|---|---|---|---|---|---|
| SimpleCIL | 81.26 (TUNA T1) | 54.55 (TUNA T1) | 48.91 (TUNA T1) | 73.15 (EASE T1) | 84.38 (EASE T1) |
| RanPAC | 90.62 (TUNA T1) | 77.94 (TUNA T1) | 61.82 (TUNA T1) | — (no source with this protocol; the RanPAC paper uses others) | — (same) |
| EASE | 85.80 (TUNA T1) | 76.17 (TUNA T1) | 55.04 (TUNA T1) | 74.85 (EASE T1) | 93.55 (EASE T1) |
| MOS | 89.25 (MOS T1) | 77.93 (MOS T1) | 59.12 (MOS T1)† | 80.05 (MOS T1) | 92.79 (MOS T1) |
| TUNA | 90.74 (TUNA T1) | 79.42 (TUNA T1) | 64.78 (TUNA T1) | — (no source with this protocol) | — (no source with this protocol) |

Notes:
- The EASE paper reports ImageNet-R with Inc5; therefore EASE's ImageNet-R number comes from the TUNA table, which uses Inc20.
- The published AAC of each method is taken from the same tables.
- **Rule:** when the method's own paper reports the number under the same protocol, that number is used; otherwise, the one from the most recent table with the same protocol. Discrepancies between sources are noted here and in the paper.
- MOS values checked on 23 Nov 2026 in Table 1 of the MOS paper (arXiv HTML version), whose columns are, for each dataset, Ā followed by 𝒜_B (final accuracy).
- † **Discrepancy:** the MOS paper reports 59.12% on ImageNet-A; the TUNA table reports 56.22% for MOS under the same protocol. We use MOS's own number and mention the difference.
- ObjectNet appears in the MOS (Inc10) and TUNA (Inc20) tables with the same values for EASE and MOS, which suggests one of the labels is wrong; it was excluded for another reason (license).
- RanPAC does not appear in the main MOS table.

---

## References cited

- ObjectNet. License, distributed with the dataset (copy at huggingface.co/datasets/timm/objectnet, LICENSE file), accessed 22 Nov 2026.
- Goswami, D. et al. FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning. NeurIPS, 2023.
- Sun, H.-L.; Zhou, D.-W.; Zhan, D.-C.; Ye, H.-J. PILOT: A Pre-Trained Model-Based Continual Learning Toolbox. SCIENCE CHINA Information Sciences, 68(4):147101, 2025.
- McDonnell, M. D. et al. RanPAC: Random Projections and Pre-trained Models for Continual Learning. NeurIPS, 2023.
- Sun, H.-L. et al. MOS: Model Surgery for Pre-Trained Model-Based Class-Incremental Learning. AAAI, 2025.
- Wang, Y. et al. Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning (TUNA). ICCV, 2025.
- Xu, L.; Zhou, Z.; Ji, W.; Miao, C.; Zhao, P.; Zhang, L. Distribution-Conditioned Task Routing for Class-Incremental Learning. arXiv 2609.34503v1, submitted 28 Sep 2026 (checked 23 Nov 2026). Standardization is tested in their Section 3.2 and Table 1: with a shared ViT-B/16 encoder and cumulative LoRA increments, accuracy changes by +0.35, −0.02, −0.79, −0.54 and +0.11 pp on CIFAR-100, ImageNet-A, ImageNet-R, CUB-200 and OmniBenchmark.
- Zhang, G. et al. SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model. ICCV, 2023.
- Zhou, D.-W. et al. Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning (EASE). CVPR, 2024.
- Zhou, D.-W. et al. Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need (SimpleCIL/APER). IJCV, 2024.
