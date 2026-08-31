# Capstone Report — CTR / Engagement Opportunity Scoring

- **Author:** Amjad Alami
- **Lane:** CTR / Engagement Opportunity Scoring
- **Repo:** https://github.com/T0othIess/FlyRank-AI-ML-internship
- **Date:** 2026-08-31

## 0. Abstract

This project asks which published pages receive lower click-through rate (CTR) than expected for their search position, and which pages an SEO team should review first. I used March 2026 data from the FlyRank AI/ML Internship warehouse, aggregated to one row per `client_hash_id × content_hash_id`, and kept pages with at least 200 monthly impressions. I built a logistic regression model using pre-click search context and a tier-relative high-gap label whose expected CTR and threshold are recalculated from training clients only. Across 50 client-grouped holdouts, the model averaged **85.00% Precision@20** compared with a **34.65% random/base-rate mean**, while the label-informed `ctr_gap` baseline reached **100.00% Precision@20**. The final output is a ranked review queue with reason codes for human SEO review; it is decision-support, not proof that changing a page will improve CTR.

## 1. Problem framing

The research question is:

**Of the published pages that have Google Search Console data available, which pages have lower CTR than expected for comparable search positions, and which pages should be prioritized for review?**

The decision is a **ranking decision**. An SEO specialist or content manager cannot review every page at once, so the useful output is an ordered queue showing which pages appear most worth checking first.

The final unit of analysis is one **`client_hash_id × content_hash_id` pair aggregated across March 2026**. The output is a model score and ranked review queue, not an automatic edit recommendation.

A human reviewer uses the queue to decide where to spend review time. Typical checks can include the title, meta description, search-intent match, and whether the content is still current.

The cost of a wrong call is mostly **wasted review time and effort**. A false positive can send an SEO reviewer toward a page that is actually performing normally, while a missed page can delay attention to a real CTR opportunity.

Data/ML helps because CTR is strongly related to search position, while the review decision also has other available context such as search volume and competition level. A fixed rule is transparent, but a model can combine several signals and rank many eligible pages consistently.

This project does **not** claim to predict Google's ranking algorithm or prove that any specific page edit causes CTR to improve.

## 2. Data safety

### Data used

The project uses the FlyRank AI/ML Internship warehouse on Hugging Face.

The main tables are:

- `fact_content_daily_performance`
- `dim_content`

The development window is **March 2026** (`month=2026-03`). June 2026 is the sealed final month in the warehouse and is not used to build or tune the model rules. I also did not use `fact_content_query_90d` because its 90-day window reaches into June.

The March daily-performance table contains **9,841,378 rows**, with **3,611,061 rows** having GSC data available, or **36.69% coverage**.

After aggregating March to client-page level, the dataset contained **159,054 rows**. I then required at least **200 total impressions**, leaving **82,507 eligible pages**.

### Feature / label / context / excluded fields

**Model features:**

- `weighted_avg_position`
- one-hot encoded `position_tier`
- `log_search_volume`
- `competition_level_enc`

**Outcome / label-building fields, not model features:**

- `total_clicks`
- `total_impressions`
- `ctr`
- `expected_ctr`
- `ctr_gap`
- `gap_threshold`
- `is_high_gap`

**Context / grouping only:**

- `client_hash_id`
- `content_hash_id`

`client_hash_id` is used to keep clients separated during validation. `content_hash_id` is used to identify a page in the queue. Neither hashed ID is a model feature.

### Leakage risks considered

The main leakage problem found during the project was in the **old label design**.

Originally, one flat threshold (`ctr_gap >= 0.002`) was used for every position tier, and `expected_ctr` was calculated before the split. The audit showed two problems:

1. the lowest-position tier had an expected CTR below `0.002`, so it could never be labeled high-gap;
2. test rows influenced the expected CTR used to construct the label.

The final capstone fixes this by recalculating `expected_ctr` and the tier-specific 70th-percentile threshold from **training rows only for every grouped split**.

Clicks, impressions, CTR, expected CTR, CTR gap, and the final label are never passed into the logistic regression as features.

### Public-safety check

The public work must not contain client names, domains, raw URLs, private queries, credentials, or raw private exports. The queue uses hashed content IDs rather than client-identifying fields.

## 3. Baseline

The original transparent baseline is:

**rank pages by `ctr_gap = expected_ctr - observed_ctr`, largest gap first.**

The reasoning is simple: a larger positive gap means the page receives fewer clicks than the benchmark for its search-position tier.

Earlier baseline work exposed an important weakness: the highest-ranked pages often had only 1–2 impressions, so a single click could completely change their CTR. That led to the final **200-impression minimum**.

For the final evaluation, the `ctr_gap` baseline is rebuilt on the same test rows and measured with the same Precision@k metric as the model.

It is intentionally a very strong baseline because it directly observes `ctr_gap`, which is also used to construct `is_high_gap`. Therefore it should be treated as a **label-informed upper reference**, not as a realistic pre-click deployable model.

I also report the **random/base rate** next to every Precision@k result. Across the final 50 grouped holdouts, the random/base-rate mean is **34.65%** and ranges from **26.46% to 45.02%** depending on which clients are in test.

Final mean baseline results:

| k | Random / base rate | `ctr_gap` baseline |
|---:|---:|---:|
| 20 | 34.65% | 100.00% |
| 50 | 34.65% | 100.00% |
| 100 | 34.65% | 100.00% |
| 200 | 34.65% | 100.00% |
| 500 | 34.65% | 99.49% |
| 1000 | 34.65% | 97.48% |

The comparison is fair in the sense that all three rankings are evaluated on the **same held-out rows with the same metric**, but the `ctr_gap` baseline has much more outcome information than the logistic regression. The random/base-rate reference is therefore the more realistic naive comparison for the learned model.

## 4. Model / analysis

### Method

The final model is:

`LogisticRegression(max_iter=1000, random_state=1)`

I kept logistic regression because it is simple, interpretable, and produces a continuous class-1 score that can be used directly for ranking.

### Exact feature list

The model uses:

- `weighted_avg_position`
- `position_tier_page_1`
- `position_tier_striking`
- `position_tier_page_3_plus`
- `log_search_volume`
- `competition_level_enc`

`position_tier` is created from average search position:

- `page_1`: position 0–10
- `striking`: position 10–20
- `page_3_plus`: position above 20

`search_volume` is transformed with `np.log1p()` because it is strongly right-skewed.

`competition_level` is encoded as:

- `LOW = 0`
- `MEDIUM = 1`
- `HIGH = 2`

### Target / proxy

The target is a **defined proxy**, not a directly observed business label.

In one sentence:

> A page is labeled `is_high_gap = 1` when its observed CTR gap is at or above the training-data 70th percentile for its own position tier.

For each split:

`expected_ctr = training clicks in tier / training impressions in tier`

then:

`ctr_gap = expected_ctr - observed_ctr`

The expected CTR and the 70th-percentile threshold are rebuilt from training rows only.

### What was deliberately left out

The model does not use clicks, impressions, observed CTR, expected CTR, CTR gap, the threshold, or the label itself. Those fields would make the model read information too close to the answer it is being evaluated against.

GA4/session/scroll fields and AI-referral fields are also excluded because the capstone is focused specifically on Google Search CTR and pre-click review prioritization.

## 5. Evaluation

### Split design

The final evaluation uses `GroupShuffleSplit` with `client_hash_id` as the grouping variable.

That means every page from a client is placed entirely in either train or test for a given split. The final code also asserts that the number of clients shared between train and test is zero.

Client sizes are highly uneven. Because `test_size=0.2` refers to approximately 20% of the **client groups**, not 20% of rows, the actual number of test rows changes substantially depending on which clients are held out.

Earlier work tried selecting seeds whose row split looked close to 80/20, but that was rejected because doing so can systematically avoid putting very large clients in test.

The final result therefore uses **50 unfiltered grouped holdouts, seeds 1–50**. For every seed, the expected CTR, thresholds, labels, logistic regression, and test metrics are rebuilt from that split.

### Metric

The main metric is **Precision@k**.

This matches the business decision because an SEO reviewer has limited review capacity and cares most about the quality of the first pages in the queue.

### Final results

| k | Random mean | `ctr_gap` baseline mean | Model mean | Model min | Model max |
|---:|---:|---:|---:|---:|---:|
| 20 | 34.65% | 100.00% | **85.00%** | 45.00% | 100.00% |
| 50 | 34.65% | 100.00% | **84.80%** | 52.00% | 98.00% |
| 100 | 34.65% | 100.00% | **83.12%** | 47.00% | 98.00% |
| 200 | 34.65% | 100.00% | **79.85%** | 42.50% | 97.50% |
| 500 | 34.65% | 99.49% | **71.89%** | 33.40% | 94.80% |
| 1000 | 34.65% | 97.48% | **63.76%** | 28.20% | 91.20% |

The model stays above the **34.65% mean base rate** at every tested `k`, with its strongest average performance at the top of the queue.

The large min/max range is important. At `k=20`, the model ranges from **45% to 100%** depending on which clients are held out. I therefore report the mean and range instead of choosing one favorable seed.

### Short error analysis

The final queue is built by saving only test-row predictions from the 50 grouped evaluations and averaging a page's score when it appears in test multiple times. This gives **100% eligible-page coverage with at least one out-of-sample prediction**.

In the final combined top-20 queue:

- **15 of 20** pages have a majority high-gap label;
- **5 of 20** are `MODEL_SCORE_ONLY_NOT_HIGH_GAP` verification cases.

Those five rows are useful errors to keep visible: the model ranks them highly, but most of the repeated outcome-based labels do not confirm them as high-gap.

The top 20 are also all in the `page_3_plus` tier. This shows that the learned ranking still concentrates heavily on deeper-ranking pages, so the queue should not be treated as a perfectly calibrated ordering across every position range.

No sealed June evaluation is claimed in this report. June remains unused.

## 6. Interpretation

The most important finding was not just a model score; it was the **label and validation audit**.

The original Week-5 model had extreme position-tier coefficients:

- `page_1`: about `+1.98`
- `striking`: about `+1.96`
- `page_3_plus` under the old name: about `-6.91`

Removing position tier from that old model collapsed the ranking toward the base rate, which exposed that the model was mostly reading back the way the label had been constructed.

After correcting the label to use train-only tier benchmarks and tier-specific thresholds, the diagnostic model's coefficients became much smaller:

| Feature | Corrected diagnostic coefficient |
|---|---:|
| `position_tier_page_3_plus` | -0.845 |
| `position_tier_striking` | -0.367 |
| `position_tier_page_1` | -0.060 |
| `competition_level_enc` | -0.018 |
| `weighted_avg_position` | +0.051 |
| `log_search_volume` | +0.094 |

That is a much less extreme pattern than the old label produced.

A useful negative result is that, on the corrected diagnostic split, removing the one-hot `position_tier` columns did **not** destroy Precision@20: both the full model and the no-position-tier version measured 95% on that one split. I do not treat that single split as the final result, but it supports the conclusion that the corrected model is no longer getting nearly all of its apparent skill from the old tier-label construction.

The strongest remaining issue is **client heterogeneity**. Some clients contain far more eligible pages than others, so a grouped test population can look very different from seed to seed. This is why the honest final result is the repeated-holdout mean plus the full observed range.

There is also a label-tie limitation. In the diagnostic training split, the intended 70th-percentile rule flagged about:

- 30.0% of `page_1`;
- 33.0% of `striking`;
- 44.3% of `page_3_plus`.

The `page_3_plus` overshoot occurs because **6,990 training pages** share the same CTR gap at the percentile boundary. Since the rule uses `>=`, the whole tie is included. I leave this visible instead of adding a more complicated tie-breaking system to the internship capstone.

Overall, the measured result is best interpreted as: **the model enriches the top of a human review queue above random selection, but the exact precision depends heavily on which clients are unseen, and the label itself is a practical heuristic rather than ground truth.**

## 7. Recommendation

The final output is a ranked action queue built from out-of-sample predictions.

For each of the 50 grouped splits, only test-row predictions are saved. If a page appears in test more than once, its model score and label-related values are averaged. The final queue is sorted only by mean `predicted_high_gap`.

The reason code adds context; it does **not** change the model ranking.

The final reason codes are:

- `HIGH_PRIORITY_CTR_GAP` — majority high-gap with higher search demand;
- `CTR_GAP_REVIEW` — majority high-gap with lower search demand;
- `MODEL_SCORE_ONLY_NOT_HIGH_GAP` — model ranked the page highly, but the majority label did not confirm high-gap, so verify first.

In the final top 20:

- 12 pages are `HIGH_PRIORITY_CTR_GAP`;
- 3 pages are `CTR_GAP_REVIEW`;
- 5 pages are `MODEL_SCORE_ONLY_NOT_HIGH_GAP`.

A FlyRank editor or SEO reviewer should start at the top of the queue and inspect the page before acting. Useful checks include:

- title and meta description;
- search-intent match;
- whether the content is still current;
- whether the CTR pattern still holds in newer data;
- whether the page is valuable enough to justify an edit.

The queue should **not** auto-publish title changes, rewrite content automatically, or treat a high model score as proof that a specific change will improve CTR.

Confidence is therefore **directional**. The model's mean top-of-queue precision is encouraging, but the 45%–100% Precision@20 range and the five verification cases in the combined top 20 show why human review is still required.

For monitoring, the Week-7 playbook suggested rerunning after roughly **60–90 days**, or earlier if search performance changes materially. On a new run, expected CTR and thresholds should be recalculated rather than reusing March values.

## 8. Reproducibility

### Fresh-clone setup

```bash
git clone https://github.com/T0othIess/FlyRank-AI-ML-internship.git
cd FlyRank-AI-ML-internship
python -m pip install -r requirements.txt
```

The full warehouse is gated, so set a Hugging Face read token before running the capstone.

Linux/macOS:

```bash
export HF_TOKEN="your_huggingface_read_token"
```

Windows PowerShell:

```powershell
$env:HF_TOKEN="your_huggingface_read_token"
```

Then open:

```text
work/notebooks/capstone.ipynb
```

and run the notebook from top to bottom.

### Main environment

The final notebook was run with Python 3.14.7.

The main packages used by the capstone are:

- `duckdb`
- `huggingface_hub`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `IPython` / Jupyter display utilities

No capstone-specific model library beyond the repository's normal Python requirements is needed.

### Random seeds

The final repeated evaluation uses:

- `GroupShuffleSplit` seeds **1 through 50**
- `LogisticRegression(random_state=1, max_iter=1000)`

There is also an earlier diagnostic grouped split chosen from a seed search in the notebook. That split is kept for readable inspection only; the **final reported metrics do not come from selecting that seed**.

### Files produced by the capstone

The final notebook produces:

- `work/outputs/action_queue_final.csv`
- `work/outputs/capstone_precision_summary.csv`
- `work/outputs/capstone_top100_review_queue.csv`
- `work/outputs/capstone_precision_at_k.png`

The main reproducible source is:

- `work/notebooks/capstone.ipynb`

The weekly notebooks under `work/notebooks/` document the progression from research question → task framing → data contract → baseline → model → validation audit → action playbook.

No sealed June holdout result is claimed, so there is no "evaluated once, blind" June metrics file to reproduce.

## 9. Acknowledgments & data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai).

This capstone was completed as part of the FlyRank AI/ML Internship using the gated internship warehouse. Public results are kept anonymized and are framed as observed, measured, directional, and decision-support evidence rather than causal proof.

---