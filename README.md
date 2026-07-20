# Assessing the Viability of Patch-Wise Robust Principal Component Analysis (RPCA) for Language Diarization

**Abstract**  
This report details the mathematical formulation, experimental evaluation, and ultimate failure diagnosis of using Patch-Wise Robust Principal Component Analysis (RPCA) for Language Code-Switching Detection. While RPCA theoretically isolates structural anomalies via sparse matrix decomposition, empirical evidence demonstrates its profound vulnerability to acoustic silence (the "Silence Confounder") and fluctuating noise topographies. Despite employing Multi-Scale Fusion and custom Voice Activity Detection (VAD) algorithms to achieve 60% accuracy on single-switch isolated audio, continuous multi-switch evaluation proves that the sparse cosine similarity signal is mathematically incapable of differentiating shallow phonotactic transitions from random acoustic noise. We conclude that language diarization fundamentally requires explicit temporal sequence modeling (e.g., Phoneme Transition Matrices) rather than structural sparsity.

---

## 1. Methodological Intuition

### The Problem with Global RPCA
Traditional RPCA operates on a signal matrix and attempts to decompose it into a low-rank background ($L$) and a sparse anomaly matrix ($S$). When applied globally to an entire multi-lingual audio file (represented as a matrix of phoneme posterior probabilities), the algorithm fails. Code-switched audio—especially between structurally distant families such as Indo-Aryan and Germanic—shares too little phonetic overlap to formulate a unifying low-rank background. Consequently, the solver forcibly distributes entire linguistic segments into the sparse matrix as chaotic noise, destroying localized anomaly detection capabilities.

### The Patch-Wise Solution
To resolve this, we employ a **Patch-Wise** methodology. By running RPCA on localized, short-duration windows (e.g., $w \leq 4.0\text{s}$), we isolate segments that are overwhelmingly monolingual. The RPCA solver can successfully adapt to the localized phonetic grammar, extracting a clean, language-specific sparse footprint. A code-switch is then hypothesized as the boundary where the structural geometry of the left patch's sparse footprint maximally diverges from the right patch's sparse footprint.

---

## 2. Mathematical Formulation

Let $M \in \mathbb{R}^{T \times K}$ represent the input phoneme posterior matrix, where $T$ is the total frame count and $K = 392$ represents the dimensionality of the phonetic vocabulary space.

### Step 1: Sliding Window Extraction
We evaluate a continuous boundary hypothesis $t$ across the temporal axis with a stride of $s$ seconds. For a given boundary $t$ and window duration $w$, we extract two independent, adjacent matrices:
1. **Left Patch ($X_L$)**: The phoneme probabilities bounded by $[t - w, t]$.
2. **Right Patch ($X_R$)**: The phoneme probabilities bounded by $[t, t + w]$.

Where $X_L, X_R \in \mathbb{R}^{K \times T_w}$ and $T_w$ is the discrete frame count within duration $w$.

### Step 2: Localized RPCA Decomposition
We solve the Principal Component Pursuit (PCP) problem independently for both patches using the Inexact Augmented Lagrange Multiplier (IALM) method. The objective function minimizes the nuclear norm of $L$ and the $L_1$ norm of $S$:

$$ \min_{L_L, S_L} \|L_L\|_* + \lambda \|S_L\|_1 \quad \text{subject to} \quad X_L = L_L + S_L $$
$$ \min_{L_R, S_R} \|L_R\|_* + \lambda \|S_R\|_1 \quad \text{subject to} \quad X_R = L_R + S_R $$

Here, $\lambda = 1 / \sqrt{\max(K, T_w)}$ is the theoretical weighting parameter determining the sparsity constraint.

### Step 3: Sparse Fingerprint Generation ($L_1$ Aggregation)
To structurally compare the sparse matrices, we collapse them across the temporal dimension $T_w$, generating a 1D structural fingerprint vector for each patch by computing the mean absolute energy (L1 Norm):

$$ v_{L}[k] = \frac{1}{T_w} \sum_{j=1}^{T_w} |S_{L}[k, j]| \quad , \quad v_{R}[k] = \frac{1}{T_w} \sum_{j=1}^{T_w} |S_{R}[k, j]| $$

This yields two vectors, $v_L, v_R \in \mathbb{R}^{K}$, representing the localized sparse phonetic distribution.

### Step 4: The Cosine Similarity Metric
The structural deviation between the two phonetic regimes is measured via the inner product space using Cosine Similarity:

$$ \text{Sim}(t) = \frac{v_L \cdot v_R}{\|v_L\|_2 \|v_R\|_2} $$

The objective is to find the global minimum of this similarity contour, representing the point of maximum structural divergence:
$$ t^* = \arg\min_{t} \text{Sim}(t) $$

---

## 3. The Fundamental Mathematical Flaw: The Loss of Phonotactics

Before examining empirical failures, it is critical to address why RPCA is mathematically inadequate for modeling structural language grammar. Language is inherently sequence-driven; the rules governing how phonemes transition into one another (Phonotactics) define a language. However, the Patch-Wise RPCA framework suffers from two fatal mathematical limitations that completely blind it to these sequential rules.

### 1. RPCA is Sequence-Agnostic (The "Bag-of-Vectors" Problem)
The objective function of Principal Component Pursuit ($\min \|L\|_* + \lambda \|S\|_1$) evaluates the column space of the input matrix $M$ based strictly on linear dependence. It treats the temporal columns (frames) as an unordered set of vectors in $\mathbb{R}^K$. 

Mathematically, if we apply a random permutation matrix $P$ to shuffle the temporal columns of our acoustic input ($M_{shuffled} = M P$), the resulting RPCA decomposition will be exactly $L P + S P$. The solver is completely invariant to temporal ordering. Because RPCA ignores the Markovian transition probabilities between adjacent frames, it is structurally incapable of modeling phonotactic sequences (e.g., differentiating between the valid sequence `/s/-/t/-/r/` and the invalid sequence `/t/-/s/-/r/`).

### 2. The Obligatory Time-Collapse (Averaging Posteriors)
Because the exact temporal alignment of phonemes within the Left Patch ($S_L$) will never identically align with the phonemes in the Right Patch ($S_R$), computing a direct matrix-to-matrix Frobenius inner product ($\langle S_L, S_R \rangle_F$) is mathematically meaningless. Two identical phrases spoken at slightly different times within the windows would yield a near-zero correlation.

To achieve **Shift-Invariance** (allowing us to compare the phonetic structure regardless of when the phonemes occurred in the 2-second window), we are mathematically forced to collapse the temporal dimension $T_w$ via the $L_1$ norm (as defined in Step 3). 

**The Fatal Consequence:** 
By collapsing the matrix into a 1D vector $v \in \mathbb{R}^K$, we reduce the sophisticated acoustic model into a rudimentary "Bag-of-Phonemes" histogram. The Cosine Similarity metric ($\text{Sim}(t)$) is no longer comparing the grammatical structure of two languages; it is merely comparing the raw frequency counts of isolated phonemes. Because a histogram cannot encode structural phonotactics, the RPCA's similarity contour fluctuates wildly based on localized phonetic utterances (acoustic noise) rather than actual grammatical language shifts.

---

## 4. Experimental Methodology

To empirically evaluate the mathematical formulation, we constructed rigorous simulation datasets designed to isolate language transitions from acoustic confounds.

### Feature Extraction
Audio waveforms were processed through a pre-trained **Wav2Vec2.0** acoustic model (specifically fine-tuned for multilingual phoneme recognition). The model extracts a frame-wise probability distribution over 392 distinct phonemes (the matrix $M$) at a temporal resolution of 20 milliseconds per frame (50Hz).

### Dataset Simulation
Because real-world code-switched audio contains uncontrollable variables (overlapping speech, extreme background noise, undetermined true boundaries), we synthetically generated strictly controlled evaluation datasets. Using monolingual utterances sourced from the **ReadSpeech** corpus, we concatenated discrete language segments (e.g., an English utterance appended seamlessly to a Hindi utterance). This guaranteed:
1. **Absolute Ground Truth:** The code-switch boundary is mathematically exact to the millisecond.
2. **Grammatical Purity:** The segments on either side of the boundary are guaranteed to be 100% monolingual, ensuring the sparse fingerprints extracted by the RPCA patches are structurally pure.

### Evaluation Regimes
We tested the algorithm under two distinct regimes:
1. **Single-Switch Evaluation:** Short audio clips (15-30s) containing exactly one code-switch. The evaluation objective is to find the global minimum similarity. Performance is measured using Mean Absolute Error (MAE) and accuracy tolerances ($\pm 1.0\text{s}$, $\pm 1.5\text{s}$).
2. **Continuous Multi-Switch Evaluation:** Long-form audio (5 minutes) containing 6 to 16 dynamic code-switches. The objective is to evaluate real-world continuous diarization using local peak detection. Performance is measured using Precision, Recall, and F1-Score.

---

## 5. The Silence Confounder (Mathematical Proof of Failure)

When evaluated on dense, VAD-trimmed speech (containing only Within-Family Indian languages), this algorithm achieved an unprecedented **76.2% accuracy**. However, when mapping the algorithm back to the original raw audio timeline, accuracy catastrophically collapsed to **18.0%**. 

**Mathematical Diagnosis:** If a patch contains a natural speech pause (silence), the acoustic model collapses its probability mass onto the `<pad>` or `<sil>` token. As a result, the phonetic distribution $X$ becomes a uniform, low-rank structure. The RPCA solver absorbs this entirely into $L$, leaving an empty or highly degenerate Sparse matrix $S \to \mathbf{0}$. 

Consequently, the norm $\|v_R\|_2 \to 0$. The Cosine Similarity between a speech patch ($v_L$) and a silence patch ($v_R$) instantly drops to near zero. The objective function $t^* = \arg\min_t \text{Sim}(t)$ mathematically optimizes for the most prominent silence in the audio file rather than the phonetic code-switch.

---

## 6. Mitigation Strategy A: The Window Size Sweep

To test if RPCA's inherent robustness could overcome the Silence Confounder without requiring VAD preprocessing, we swept the window size $w$ from 1.0s to 5.0s on the raw audio timeline.

### Raw Audio Evaluation
| Window (s) | Mean Absolute Error (s) | Acc @ 1.0s | Acc @ 1.5s |
| :---: | :---: | :---: | :---: |
| 1.0 | 4.790 | 12.0% | 22.0% |
| 2.0 | 4.100 | 24.0% | 32.0% |
| 3.0 | 2.940 | 36.0% | 46.0% |
| 4.0 | 2.010 | 48.0% | 60.0% |
| **5.0** | **1.939** | **49.0%** | **65.3%** |

**Interpretation:** The theoretical robustness of the $L_1$ norm constraint allows RPCA to ignore small sparse corruptions. If a 2-second window contains a 1-second pause, 50% of the patch is corrupted. However, in a 5-second window, the remaining 4 seconds (80%) provide enough data for the solver to construct a stable $L$ matrix and extract the true language-specific $S$ footprint, effectively ignoring the silence entirely.

---

## 7. Mitigation Strategy B: The VAD Resolution Tradeoff

We conducted an identical sweep on the pure **VAD-trimmed timeline** to determine the absolute optimum temporal window required to capture a language's structural grammar.

### VAD Audio Evaluation
| Window (s) | Mean Absolute Error (s) | Acc @ 1.0s | Acc @ 1.5s |
| :---: | :---: | :---: | :---: |
| 1.0 | 4.166 | 36.0% | 48.0% |
| 2.0 | 4.015 | 39.6% | 45.8% |
| 3.0 | 2.933 | 51.1% | 55.6% |
| **4.0** | **2.508** | **53.5%** | **60.5%** |
| 5.0 | 2.012 | 48.7% | 59.0% |

**Interpretation:** The accuracy follows a distinct parabolic trajectory peaking at **4.0 seconds**.
- **Under-parameterized (1-2s):** The window lacks sufficient phonetic diversity to formulate a stable grammatical fingerprint.
- **Over-parameterized (5s):** The window overshoots the code-switch boundary, forcing the patch to encapsulate a multi-lingual mixture (Language A + Language B), which degrades the sparsity constraint and weakens the similarity divergence.

---

## 8. Multi-Scale Fusion and Energy VAD

When a speaker pauses for 1.5s to switch languages, traditional VAD forcibly splices the two languages together. While the boundary is mathematically exact on the dense VAD timeline, mapping it back to the original real-time timeline introduces an irreducible error (placing the prediction at the edge of the silence rather than the center).

To combat this, we developed a **Multi-Scale Fusion Strategy**:
1. **Coarse Pass ($w=4.0\text{s}$):** Resolves the general candidate region on the dense timeline.
2. **Fine Pass ($w=2.0\text{s}$):** A high-resolution scan ($\pm 2.0\text{s}$ search radius) to precisely snap to the transition point.

Furthermore, empirical testing revealed that generic VAD models (e.g., Silero) were overly aggressive, deleting micro-pauses and up to 51% of the audio, destroying the local phonetic sequence. We replaced this with a custom **Energy-Based VAD**, strictly thresholding RMS energy to remove only genuine silences $>0.5\text{s}$ (reducing files by a safe 4-10%).

### Mapped Results
| Preprocessing Method | Mean Absolute Error (s) | Acc @ 1.5s |
| :--- | :--- | :--- |
| **Silero VAD (Mapped)** | 3.212 | 44.2% |
| **Raw Audio (No VAD)** | 2.002 | 57.1% |
| **Energy VAD (Mapped)**| **2.504** | **56.0%** |

---

## 9. The Structural Distance Hypothesis (English-Indic Evaluation)

To validate the algorithmic behavior across structurally disjoint language families, we generated a targeted dataset (`test_samples_eng_indic`) containing 50 simulated samples where every file consists of exactly one English utterance and one Indic utterance.

### Targeted Evaluation Results
| Metric | Random Languages (Prev) | English-Indic (New) |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | 2.504s | **2.391s** |
| **Accuracy @ 1.0s** | 42.0% | **52.0%** |
| **Accuracy @ 1.5s** | 56.0% | **60.0%** |
| **Accuracy @ 2.0s** | 58.0% | **60.0%** |

**Observation:** The Multi-Scale RPCA performs substantially better (+4.0% Acc @ 1.5s) when evaluating English-Indic transitions compared to intra-family Indic transitions. 

**Mathematical Intuition:** English phonetic grammar is highly orthogonal to Indic phonotactics. The transition between these domains creates a significantly deeper divergence in the $L_1$ aggregated sparse feature space than a transition between structurally similar languages (e.g., Hindi and Punjabi).

---

## 10. Continuous Multi-Switch Evaluation (The Final Proof)

While the algorithm performed adequately (60.0% Acc) on isolated, single-switch audio (where the objective was merely isolating the global minimum $\arg\min_{t} \text{Sim}(t)$), the true standard of Language Diarization is continuous audio containing multiple, dynamic boundaries.

We generated a 5-minute Multi-Switch Dataset (20 files, 6 to 16 code-switches each). To detect boundaries dynamically, we applied **Topographic Prominence** (`scipy.signal.find_peaks`), searching for local similarity dips with a prominence of `0.08`.

### Multi-Switch Results
- **Precision:** 11.1%
- **Recall:** 13.8%
- **F1-Score:** 12.3%

### The "Global vs Local" Sanity Check
To ensure this failure was algorithmic and not an artifact of the dataset, we applied the dynamic prominence script back onto the original 20 short, single-switch files (which previously achieved 60% accuracy).

**Sanity Check Results (20 files):**
- **Precision:** 40.0%
- **Recall:** 30.0%

In 14 out of the 20 single-switch files, the script returned **0 True Positives and 1 False Negative**. It failed to detect the boundary entirely.

### Mathematical Failure Diagnosis
The sanity check exposed a fatal flaw in applying similarity metrics to language transitions: **Topographic Prominence requires a signal dip to recover on both sides.**

For a missed ground truth boundary (e.g., `sample_10_ben_eng`), the similarity contour was:
- **9.0s:** 0.926 (Bengali Baseline)
- **10.0s:** 0.856 (Code-Switch Dip)
- **11.0s:** 0.874 (English Baseline)

Because the new language (English) possessed a naturally lower structural similarity baseline (`0.874`) than the previous language (`0.926`), the contour never recovered to form a clean "V-shape". It instead formed a "staircase" step down. 
The mathematical prominence of the dip was calculated solely against its lowest ridge (the right side): `0.874 - 0.856 = 0.018`. Being only `0.018`, it was completely ignored by the required `0.08` threshold.

### The "One-Sided Drop" Hypothesis
To mitigate the staircase problem, we engineered a custom peak-detection algorithm. Instead of requiring bilateral recovery, this logic triggers if the similarity drops sharply from *either* the left window OR the right window ($\geq 0.05$ across a $\pm 2.0\text{s}$ neighborhood).

**One-Sided Drop Results (20 Single-Switch files):**
- **Precision:** 20.7%
- **Recall:** 30.0%
- **False Positives:** 23 (up from 9)

**Diagnosis:** Relaxing the topographic constraint to allow unilateral drops destroyed Precision. Because the similarity contour inherently lacks structural sequence modeling, it is permeated with random acoustic variance. The algorithm latched onto massive amounts of random background noise. 

---

## 10. The Global Minimum Illusion vs. Continuous Thresholding

To fully understand the failure mode of RPCA, we must mathematically differentiate why the algorithm achieved a passable **60.0% accuracy** on isolated single-switch files, yet collapsed to a **12.3% F1-Score** on continuous multi-switch audio. The discrepancy lies in the fundamental difference between evaluating a constrained Global Minimum versus dynamic Continuous Thresholding.

### The Single-Switch "Cheat" (Global Minimum)
In a 30-second single-switch file, the mathematical objective is heavily constrained: we are a priori guaranteed that exactly *one* language transition exists. Because of this guarantee, the algorithm does not need to classify whether a dip in similarity is a language shift or acoustic noise; it merely scans the entire temporal contour and selects the **absolute lowest point** ($t^* = \arg\min_t \text{Sim}(t)$). 
Even if the similarity dip caused by the code-switch is incredibly shallow, or the baseline is highly erratic, the algorithm succeeds as long as the true switch happens to be the single deepest point in that short window.

### The Multi-Switch Reality (Signal-to-Noise Ratio)
In a continuous 5-minute audio stream, the a priori constraint vanishes. We do not know if there are 2 language transitions or 20. Therefore, the algorithm must evaluate *every* local minimum in the similarity contour and independently classify it using a mathematical threshold (e.g., Topographic Prominence $> 0.08$).

This unconstrained evaluation exposes the fatally broken **Signal-to-Noise Ratio** of the RPCA structural similarity metric:
1. **Weak True Signals:** The similarity drops caused by genuine language transitions (the true signal) are often mathematically shallow (e.g., a drop of only `0.018` due to the aforementioned staircase effect).
2. **Strong Noise Signatures:** The similarity drops caused by random acoustic variance (e.g., non-speech vocalizations, erratic phonemes, micro-pauses) regularly exceed `0.03` or `0.04`.

Because the noise signatures are frequently stronger than the true linguistic signals, it is mathematically impossible to define a separating threshold boundary. A threshold strict enough to ignore the noise (e.g., `0.08`) will completely miss the true shallow transitions (destroying Recall). Conversely, a threshold lenient enough to catch the shallow transitions (e.g., `0.018`) will instantaneously latch onto hundreds of random acoustic noise fluctuations (destroying Precision).

---

## 11. Conclusion

The empirical evidence demonstrates that Patch-Wise RPCA is mathematically unsuitable for continuous Language Diarization. Because the RPCA Sparse matrix explicitly models structural sparsity rather than temporal grammar (phonotactics), the resulting similarity signal cannot differentiate between a genuine linguistic transition and random acoustic variance. There exists no mathematical threshold—whether Global Minimum, Topographic Prominence, or One-Sided Drops—capable of isolating the true signal from the noise.

Future development must pivot entirely to **Sequence-Aware Models** (such as Phoneme Transition Matrices, Markovian Modeling, or Jensen-Shannon Divergence) that explicitly encode the temporal sequences of structural grammar.

---

## 12. Epilogue: The Video Surveillance vs. Language Paradigm

To fully contextualize why an algorithm as powerful as RPCA fails so profoundly on code-switching, we must examine its origin. RPCA was originally popularized for **Video Surveillance** (e.g., removing a moving person from a fixed camera feed of a parking lot). The mathematical assumptions that make RPCA brilliant for video are the exact assumptions that make it incompatible with language.

### Spatial Dominance (The Video Paradigm)
In video surveillance, the "background" (the concrete, the lines, the buildings) is overwhelmingly dominant, static, and highly correlated across time. If you flatten video frames into columns and stack them into a matrix, the identical background pixels form an incredibly strong, perfectly linear **Low-Rank structure ($L$)**. 
When a person walks across the camera, they occupy a tiny fraction of pixels, disrupting the static consistency. RPCA easily flags these pixels as a **Sparse anomaly ($S$)**. 

Crucially, **sequence does not matter in video.** If you take a 10-minute video of a parking lot and randomly shuffle all the frames, the parking lot is still in the exact same place. RPCA does not need to know the temporal order to calculate that the concrete is always there.

### Temporal Dominance (The Language Paradigm)
Spoken language operates on entirely different physics. Language has no "static parking lot." 
When someone speaks, they do not emit a constant, unchanging hum of the exact same 5 phonemes in every single frame. The acoustic output is constantly shifting and transitioning. There is no static, spatial background for RPCA to easily lock onto as the $L$ matrix.

In language, the "background" is not a physical object; it is the **invisible grammatical rules (phonotactics)** governing how phonemes flow into one another over time. 
If you take a 10-second audio clip of speech and randomly shuffle all the frames, you completely destroy the language, rendering it unrecognizable static. But because RPCA is mathematically invariant to shuffling, it cannot see the phonotactic rules. It merely sees a chaotic, unorganized mess of phonemes.

### The Fatal Mismatch
Because RPCA is looking for a static physical background in a dataset where everything is constantly moving, it fails to find a strong Low-Rank structure. Consequently, it dumps massive amounts of actual, vital linguistic speech into the Sparse matrix as if it were chaotic noise. 

In video, an anomaly is a **spatial disruption**. In language, a code-switch is a **sequential disruption** (the rules of how phonemes follow each other suddenly change). Because RPCA is mathematically blind to sequence, it is the fundamentally incorrect tool for language modeling.
