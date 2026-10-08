# Grammar Scoring Engine for Spoken Audio

Predicts a continuous grammar score (0 to 5) from 45 to 60 second spoken audio samples.
Built for the **SHL Hiring Assessment 2026** (Kaggle).

**Approach in one line:** transcribe the audio with Whisper, turn each transcript into nine
interpretable features that mirror the grading rubric, and fit a regularised linear model,
validated with 5-fold cross-validation against a mean-prediction baseline.

```
audio (.wav) → Whisper transcript + timings → 9 rubric-based features → Ridge regression → score 0–5
```

**Result:** cross-validated RMSE **0.838** vs a baseline of 1.238 (a 32% reduction), with a
Pearson correlation of **0.736** between predicted and human scores.

---

## 1. Problem

| Item | Detail |
|---|---|
| Input | One `.wav` file per sample, 45 to 60 seconds of speech |
| Target | MOS Likert grammar score, predicted as a continuous value from 0 to 5 |
| Training data | 769 labelled samples |
| Test data | 216 samples |
| Output | `submission.csv` with columns `filename`, `label` |

**Rubric (summary).** Low scores (1 to 2) mean simple or memorised sentence patterns, basic
mistakes and incomplete sentences. Middle scores (3) mean a decent grasp of structure with
errors in grammar or syntax. High scores (4 to 5) mean consistent control, complex structures,
rare minor errors and self-correction.

Training label distribution (count per score):

| 0 | 1 | 1.5 | 2 | 2.5 | 3 | 3.5 | 4 | 4.5 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| 37 | 1 | 3 | 102 | 79 | 174 | 64 | 124 | 52 | 133 |

The 37 samples scored 0 are likely silent or unintelligible recordings.

---

## 2. Method

### 2.1 Transcription
Each file is transcribed with OpenAI **Whisper (`small`)** on a GPU.

- Audio is loaded and resampled to 16 kHz mono.
- Word-level timestamps are requested, which give pause and speaking-rate information.
- Whisper tends to "clean up" speech and hide hesitations and repetitions. A short
  `initial_prompt` containing fillers and repeated words nudges it to keep them, because the
  rubric rewards self-correction and penalises incomplete sentences.
- Results are cached to disk so transcription runs only once.

### 2.2 Features
Every feature is tied to a line of the rubric.

| Feature | What it measures | Rubric link |
|---|---|---|
| `grammar_errors_per_100w` | Grammar errors found by LanguageTool, per 100 words | "consistently make grammatical mistakes" |
| `avg_sentence_len` | Words per sentence | simple fragments vs developed sentences |
| `vocab_diversity` | Unique words ÷ total words | "memorised sentence patterns" |
| `fillers_per_100w` | "um", "uh", "hmm" and similar | hesitation / fluency |
| `repeats_per_100w` | Immediately repeated words ("I I went") | restarts and self-correction |
| `n_words` | Amount of speech produced | context for the rates above |
| `words_per_min` | Speaking pace | fluency |
| `pause_fraction` | Share of time spent in pauses longer than 0.5 s | fluency |
| `asr_confidence` | Whisper's average log-probability | unclear or broken speech |

### 2.3 Models and validation
- **Ridge regression** (standardised inputs, regularisation strength chosen by internal CV) and
  a **Random Forest** are compared with **5-fold cross-validation** on the training set.
- Metrics: RMSE, MAE and Pearson correlation, compared with a baseline that always predicts
  the training mean.
- The model with the lower cross-validated RMSE (Ridge) is refit on all training data and used
  for the test set.
- Predictions are clipped to [0, 5] and **not rounded**, since the target is continuous.

---

## 3. Results

Cross-validated performance on the 769 training samples (5-fold):

| Model | RMSE | MAE | Pearson r |
|---|---|---|---|
| Baseline (always predict the mean) | 1.238 | – | – |
| Random Forest | 0.883 | 0.689 | 0.701 |
| **Ridge regression (selected)** | **0.838** | **0.645** | **0.736** |

Ridge reduces the error by about 32% relative to the baseline and predicts the human score to
within roughly 0.65 points on average on the 0 to 5 scale. The simpler linear model outperformed
the Random Forest, so the more interpretable model was also the better one. Test labels are not
available, so the cross-validated figures above are the performance estimate.

### Which features matter

| Feature | Correlation with score | Ridge coefficient (standardised) |
|---|---|---|
| `asr_confidence` | +0.57 | +0.735 |
| `n_words` | +0.28 | +0.695 |
| `words_per_min` | +0.24 | −0.329 |
| `vocab_diversity` | +0.22 | +0.595 |
| `fillers_per_100w` | −0.05 | +0.029 |
| `repeats_per_100w` | −0.08 | +0.004 |
| `pause_fraction` | −0.08 | +0.053 |
| `avg_sentence_len` | −0.11 | −0.054 |
| `grammar_errors_per_100w` | −0.12 | −0.081 |

Observations:

- **Speech clarity is the strongest signal.** Whisper's confidence has by far the highest
  correlation (0.57), and speakers who produce more words with more varied vocabulary score higher.
- **The grammar-error rate has the expected sign but a weak correlation (−0.12).** Whisper
  repairs many grammar mistakes while transcribing, so the grammar checker sees fewer errors than
  were actually spoken.
- **`words_per_min` has a positive raw correlation but a negative coefficient.** It overlaps
  strongly with `n_words`, so once word count is accounted for, its coefficient flips sign.
  Individual coefficients of correlated features are therefore only indicative, and single-feature
  correlations are the safer basis for claims.

### Sanity checks on the submission

| Check | Result |
|---|---|
| Rows / columns | 216 rows, `filename` and `label` |
| Duplicate IDs / IDs missing from `test.csv` | 0 / 0 |
| Prediction range | 0.33 to 5.00 (inside 0 to 5) |
| Prediction std vs training label std | 0.71 vs 1.24 (narrower, as expected for a regression model) |
| Identical transcripts for train/test files with the same name | 0 |
| Empty transcripts | 0 |

---

## 4. Design decisions

- **Interpretable features over a black box.** With 769 samples, fine-tuning a transformer is
  prone to overfitting and hard to explain. A small set of rubric-based features can be traced
  from a prediction back to the speech.
- **Disfluencies are kept, not removed.** Self-correction is part of the top score level.
- **Honest evaluation.** Cross-validation with a baseline comparison, not training error.
- **Continuous output.** Predictions are not rounded to integers.
- **Train and test share file names.** 212 file names (for example `audio_0.wav`) appear in both
  folders and are different recordings. All caching and lookups use `(split, filename)` as the
  key so train and test are never mixed, and a check confirms no pair has identical transcripts.
- **Submission format.** `sample_submission.csv` lists IDs that do not match `test.csv`
  (only 25 of its 204 IDs appear in the test file). The submission therefore uses the 216
  filenames from `test.csv`.

---

## 5. Limitations

- Transcription errors propagate into every feature, especially for accented or noisy speech.
- Whisper partly corrects grammar while transcribing, so some real errors are never seen by the
  grammar checker. The model relies partly on `asr_confidence` as a proxy for unclear or
  broken speech, which measures intelligibility as much as grammar.
- Grammar is checked on text only. Pronunciation and prosody are not modelled.
- Labels are subjective human ratings, so some noise is unavoidable.
- Very low scores (0) are likely driven by audio quality rather than grammar.

## 6. Possible improvements

- Syntactic complexity features (clause counts, parse-tree depth, tense variety) from spaCy.
- Language-model perplexity of the transcript as a naturalness signal.
- Frozen sentence embeddings with Ridge, blended with the feature model.
- Audio-level embeddings (for example Whisper encoder features) to capture delivery.
- An ablation study to measure how much each feature group adds.

---

## 7. How to run

The pipeline was developed as a Kaggle notebook with a **GPU (T4)** and **Internet** enabled.

1. Attach the competition data and install the dependencies:
   ```
   apt-get install -y default-jre-headless
   pip install openai-whisper language-tool-python scikit-learn pandas numpy tqdm
   ```
2. Run the notebook cells in order:
   1. Install libraries
   2. Locate and verify the data
   3. Transcribe with Whisper (about 30 to 50 minutes on a T4; cached afterwards)
   4. Build features
   5. Cross-validate and choose the model
   6. Fit the final model and write `submission.csv`
   7. Sanity checks
3. The output is `submission.csv` with 216 rows and the columns `filename`, `label`.

## 8. Repository contents

```
├── Grammar Scoring Engine.ipynb   # full pipeline
├── submission.csv                 # final predictions
└── README.md
```

The audio files and competition data are not included.

## 9. Tech stack

Python, OpenAI Whisper, LanguageTool, scikit-learn (Ridge, Random Forest, cross-validation),
pandas, NumPy.
