# Capstone Report — Lane 2: Refresh / Content Opportunity Scoring

- **Author:** Muhammad Umer
- **Lane:** Lane 2 — Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/umerkang66/flyrank-ml-internship
- **Date:** September 2026

---

## 0. Abstract

How can search editorial teams managing tens of thousands of multi-client content assets prioritize decaying pages for refresh before organic visibility collapses? We investigated this question on the FlyRank Search Intelligence dataset, comprising 30,000 pseudonymized content assets across 32 client domains evaluated over a trailing 90-day observation window. We formulated the task as pointwise probabilistic ranking, training and comparing Logistic Regression, shallow Decision Trees, and a Random Forest ensemble evaluated against a deterministic heuristic baseline under a strict 80/20 grouped client split (26 training clients vs. 6 unseen holdout clients). The Random Forest model achieved 90.0% precision at Top-20 and Top-50 on unseen clients (a 1.46× lift over the 61.72% test base rate, with ROC-AUC of 0.6757 and PR-AUC of 0.7310), substantially outperforming the heuristic rule which degraded to 60.0% precision due to arbitrary tie-breaking. Error auditing confirmed that top model disagreements represent high-upside strategic opportunities (deep-page assets with zero CTR) rather than algorithmic failures, while low-volume ghost pages were correctly dampened. This operational ranking engine equips content teams with a transparent, evidence-backed review queue that maximizes the ROI of constrained editorial hours.

---

## 1. Problem Framing

### The Business Decision

Content portfolios experience continuous organic decay as search intent shifts, competitive counter-content emerges, and search engine ranking algorithms recalibrate. In an agency or enterprise setting managing tens of thousands of published URLs across diverse client domains, editorial bandwidth is strictly bounded: an editorial team can realistically review, update, or expand only 20 to 50 assets per week. The core operational decision is **triage allocation**: determining which specific URLs should receive manual editorial intervention next to protect organic traffic or recover lost search visibility.

### Unit of Analysis & System Output

- **Unit of Analysis:** A single published content asset on a client domain (`content_id × client_id`).
- **System Output:** A continuous, calibrated **Opportunity Score** ($S \in [0, 1]$ representing predicted decay probability $P(\text{declining} \mid X)$) mapped to a prioritized review queue, annotated with transparent reason codes and suggested action labels (`refresh`, `ctr_rewrite`, `expand`, `monitor`).
- **Human Action:** An SEO specialist or content editor acts directly on the ranked queue by executing targeted remediations:
  - _Content Refresh:_ Updating outdated facts, statistics, outbound links, and publication timestamps on decaying high-traffic pages.
  - _Metadata/CTR Rewrite:_ Crafting compelling title tags and meta descriptions for pages maintaining Page 1 impressions but suffering sub-baseline click-through rates.
  - _Content Expansion:_ Adding depth, comprehensive FAQs, or structured sections to thin articles slipping in average search position.
  - _Passive Monitoring:_ Leaving stable, evergreen, or low-demand assets untouched to preserve editorial hours.

### Cost of a Wrong Call

- **False Positive (Flagging a healthy or low-upside page):** Direct financial waste of editor and writer hours (costing hundreds of dollars per piece) and the operational risk of disrupting high-performing content with unnecessary alterations that could trigger temporary SERP volatility.
- **False Negative (Failing to surface a high-demand decaying flagship page):** Silent compounding loss of organic search visibility, surrender of competitive keyword market share, and severe loss of qualified organic conversions.

### Why Machine Learning Helps Beyond Heuristic Rules

Heuristic rules (such as "refresh all articles older than 180 days") suffer from fatal structural flaws. In our exploratory data audit, a static staleness rule flagged nearly 10,000 pages indiscriminately without regard to actual search exposure, position trends, or query demand. Furthermore, empirical auditing revealed an evergreen survivor bias where pages surviving beyond 180 days actually exhibited lower decline rates (46.7%) than pages in the 91–180 day window (61.1%). Supervised machine learning synthesizes multi-dimensional, non-linear interactions—combining search impressions, ranking trajectory, CTR expectations, and content depth—into a single calibrated priority score, enabling continuous ranking across the entire portfolio without arbitrary cutoff cliffs.

---

## 2. Data Safety & Leakage Prevention

### Data Source & Scope

This study utilizes the FlyRank ML Internship starter dataset (`data/raw/content_refresh_anonymized.csv`), consisting of **30,000 rows × 44 columns**, representing 30,000 unique content assets across 32 pseudonymized client sites. Each record aggregates trailing 90-day search performance (Google Search Console metrics), on-site user behavior (Google Analytics 4 metrics), and content catalog metadata.

### Deliberately Excluded Columns & Leakage Controls

To maintain strict methodological integrity, columns were partitioned and audited against target leakage:

| Excluded Column               | Category                  | Rationale for Exclusion                                                                                                                                                                         |
| :---------------------------- | :------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `trend_direction`             | **Target Proxy**          | The binary ground truth label `is_declining_label` is directly computed from `trend_direction == 'down'`. Including this column or any direct derivative is textbook label leakage.             |
| `trend_pct`                   | **Direct Target Leakage** | `trend_direction` is mathematically derived from `trend_pct` thresholds. Feeding percentage change into a model predicting directional change allows the model to trivially memorize the label. |
| `content_id`                  | **Pseudonymous ID**       | Page identifier; used strictly for tracking records, never as an input feature.                                                                                                                 |
| `client_id`                   | **Pseudonymous ID**       | Client identifier; used strictly for grouped train/test partitioning, never as an input feature.                                                                                                |
| `provider_used`, `model_used` | **Missingness Trap**      | Sparsely populated metadata with structural missingness tied to specific client CMS setups; excluded to prevent learning spurious client-level artifacts.                                       |

### Public-Safety Confirmation

In strict accordance with `DATA_USE.md` and repository guidelines:

- No client names, raw URLs, domains, unhashed search queries, or internal IP addresses appear in any code, markdown, or exported artifacts.
- All numbers reflect aggregated, anonymized metrics.
- Findings are reported in rigorous, defensible language: _observed, measured, directional, decision-support_, avoiding unproven causal claims regarding Google search ranking algorithms.

---

## 3. Baseline Heuristic Rule

### Plain-Words Formulation

Before fitting machine learning models, we established a transparent, deterministic baseline rule encoding standard SEO industry intuition:

> A content asset is flagged for editorial refresh if it has not been updated in at least 180 days and still maintains active search exposure (at least 500 impressions over the trailing 90 days). Eligible pages are ranked by descending 90-day search volume; content failing either gate is scored 0 and assigned to passive monitoring.

Mathematically:
$$\text{stale} = \mathbb{I}(\text{days\_since\_last\_update} \ge 180)$$
$$\text{visible} = \mathbb{I}(\text{impressions\_90d} \ge 500)$$
$$\text{baseline\_action\_score} = \text{stale} \times \text{visible} \times \text{impressions\_90d}$$

### Empirical Signal Audit Pre-Check

In Week 4 ([`w04_baseline_score.ipynb`](file:///D:/Workspace/internship_2026/work/notebooks/w04_baseline_score.ipynb)), we audited the underlying signals against observed decline (`is_declining_label == 1`, portfolio base rate = 54.21%):

1. **Staleness (`days_since_last_update`):** Decline rate rose from 51.1% (0–30d) to 61.1% (91–180d, $n=9,171$), but dropped to 46.7% at 181–365d ($n=169$). **Verdict: MIXED.** (Evergreen survivor bias invalidates the assumption of monotonic decay past 180 days).
2. **Search Visibility (`impressions_90d`):** Active impression tiers (100 to 5,000+) exhibited decline rates between 54.7% and 63.4%, compared to only 38.9% for low-visibility assets ($<100$ impressions, $n=8,006$). **Verdict: CONFIRMED.**
3. **Page 1 CTR (`avg_position <= 10`):** Low CTR ($\le 0.10\%$) on Page 1 showed an elevated decline rate (56.7%–66.6%) vs. 43.5% for healthy CTR ($>1.00\%$). **Verdict: CONFIRMED.**

### Baseline Performance on the Held-Out Test Split

When evaluated on the exact 4,723 test pages from 6 unseen clients (test base decline rate: 61.72%):

- **Precision@10:** 0.5000 (5/10 correct | Lift: 0.81×)
- **Precision@20:** 0.6500 (13/20 correct | Lift: 1.05×)
- **Precision@50:** 0.6000 (30/50 correct | Lift: 0.97×)
- **Precision@100:** 0.6500 (65/100 correct | Lift: 1.05×)
- **ROC-AUC:** 0.5002 | **Average Precision:** 0.6173 | **Brier Score:** 0.6172

_Flaw of the Baseline:_ The heuristic rule flagged only 17 pages total across the entire 30,000-page dataset. Beyond Rank 17, scores collapsed to 0, forcing arbitrary tie-breaking. Consequently, at $K=50$ and $K=100$, precision dropped to the random base rate floor.

---

## 4. Model Architecture & Feature Engineering

### Method Progression

We developed a tiered modeling progression to balance readability, non-linearity, and ranking power:

1. **Logistic Regression (Standardized Linear Benchmark):** Models log-odds with $L_2$ regularization and balanced class weights. Provides readable, signed directional coefficients.
2. **Decision Tree Classifier (max_depth=5, min_samples_leaf=50):** Generates transparent, readable threshold splits without manual feature scaling.
3. **Random Forest Classifier (200 trees, max_depth=10, min_samples_leaf=25):** Combines bagging and random feature sub-spacing with `balanced_subsample` weighting. Resolves complex non-linear interactions and outputs smooth, continuous probability estimates for ranked retrieval.

### Exact Feature Set (26 Numeric Features + Categorical Encodings)

- **Continuous Traffic & Engagement Signals:** `impressions_90d`, `clicks_90d`, `sessions_90d`, `ai_sessions_90d`, `days_with_impressions`, `days_with_sessions`, `ctr`, `avg_position`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`.
- **Log-Transformed Counts (Skew Reduction):** `log_impressions_90d`, `log_clicks_90d`, `log_sessions_90d`, `log_ai_sessions_90d` (via $\log(1 + \max(0, x))$).
- **Catalog & Content Structure:** `word_count`, `char_count`, `content_age_days`, `days_since_last_update`, `search_volume`, `competition`, `cpc`.
- **Categorical Encodings (One-Hot):** `competition_level`, `content_type`, `main_intent`, `age_tier`, `freshness_tier`, `word_count_tier`, `impression_tier`, `position_tier`.

### Target Proxy Definition

The model target is `is_declining_label` ($\in \{0, 1\}$), defined strictly as:
$$\text{is\_declining\_label} = \mathbb{I}(\text{trend\_direction} = \text{'down'})$$
This identifies pages where recent 30-day organic performance has dropped relative to preceding baseline velocity.

---

## 5. Evaluation & Out-of-Sample Verification

### Grouped Client Split Design

To eliminate intra-client data leakage, the dataset was partitioned using a **Grouped Client Split** (seed = 42):

- **Training Set:** 26 clients, 25,277 pages (84.3% of portfolio), base decline rate = 52.80%.
- **Test Set:** 6 held-out clients, 4,723 pages (15.7% of portfolio), base decline rate = 61.72%.
- **Client Overlap:** Strictly **0** intersecting clients. The test set evaluates whether learned signals generalize across distinct domains and CMS architectures.

### Head-to-Head Comparison Table

All strategies evaluated on the identical 4,723 held-out test rows:

| Strategy / Model             | Precision@10 | Precision@20 | Precision@50 | Precision@100 |  ROC-AUC   | PR-AUC (Avg Prec) | Brier Score | Lift@50 vs Base |
| :--------------------------- | :----------: | :----------: | :----------: | :-----------: | :--------: | :---------------: | :---------: | :-------------: |
| **Random Floor (Base Rate)** |    0.6172    |    0.6172    |    0.6172    |    0.6172     |   0.5000   |      0.6172       |   0.6172    |      1.00×      |
| **Week 4 Baseline Rule**     |    0.5000    |    0.6500    |    0.6000    |    0.6500     |   0.5002   |      0.6173       |   0.6172    |      0.97×      |
| **Decision Tree (depth=5)**  |    0.6000    |    0.4500    |    0.4800    |    0.6100     |   0.6758   |      0.7118       |   0.2227    |      0.78×      |
| **Logistic Regression**      |    0.6000    |    0.5000    |    0.7000    |    0.6700     | **0.7177** |    **0.7440**     | **0.2145**  |      1.13×      |
| **Random Forest (Ensemble)** |  **0.8000**  |  **0.9000**  |  **0.9000**  |  **0.8800**   |   0.6757   |      0.7310       |   0.2176    |    **1.46×**    |

> [!IMPORTANT]
> **Key Finding:** Random Forest achieved **90.0% precision at Top-20 and Top-50** (18/20 and 45/50 decaying assets correctly surfaced), delivering a **1.46× lift** over the test base rate. The baseline rule failed to differentiate beyond 17 assets, while the shallow tree suffered from coarse step-function probabilities at leaf nodes.

### Error Analysis: Concrete Failure Modes

A short error analysis of top mispredictions provides critical operational context:

1. **High-Score False Positive (`content_331182ca4cae`):**
   - _Signals:_ Score = 0.7699, Impressions = 3,026, Avg Position = 35.9 (Page 4), CTR = 0.00%, Age = 134d. Label = Stable (0).
   - _Why the model flagged it:_ Massive search exposure combined with deep Page 4 ranking and zero clicks strongly matches the profile of decaying assets.
   - _Operational Takeaway:_ While labeled negative by recent trailing counts, this page is failing to capture any search traffic. Editorial consolidation, pruning, or rewrites are practically justified.
2. **High-Score False Positive (`content_b15a8dbdf66f`):**
   - _Signals:_ Score = 0.7577, Impressions = 1,647, Avg Position = 22.4 (Page 3), CTR = 0.18%, Age = 144d. Label = Stable (0).
   - _Operational Takeaway:_ Striking-distance asset on a niche evergreen topic that held traffic steady despite aging copy.
3. **Low-Score False Negative (`content_28b4223f4e5f`):**
   - _Signals:_ Score = 0.0649, Impressions = 1, Avg Position = 0.0, CTR = 0.00%, Updated = 1d ago. Label = Declining (1).
   - _Why the model dampened it:_ A nominal decline from 2 impressions to 1 triggers a "down" label (-50%), but allocating editorial hours to a 1-impression page would represent catastrophic ROI waste. The model correctly assigned low priority.

---

## 6. Interpretation & Signal Mechanics

### What the Model Leans On

Feature importances from the Random Forest ensemble reveal the empirical mechanics of search decay:

| Rank | Feature                 | Importance | Plain-Words Meaning                                                                                                             |
| :--: | :---------------------- | :--------: | :------------------------------------------------------------------------------------------------------------------------------ |
|  1   | `days_with_impressions` |   10.89%   | **Search Presence Consistency:** Pages appearing daily face continuous algorithm re-evaluation and competitor counter-attacks.  |
|  2   | `avg_position`          |   10.58%   | **SERP Vulnerability:** Positions 8–25 represent high rank volatility where small algorithm shifts cause large visibility loss. |
|  3   | `impressions_90d`       |   10.18%   | **Search Exposure Demand:** High-demand topics draw aggressive competitive content creation.                                    |
|  4   | `log_impressions_90d`   |   9.67%    | Non-linear scaling of demand volume.                                                                                            |
|  5   | `content_age_days`      |   8.79%    | **Catalog Longevity:** Older content experiences natural topical and link decay.                                                |
|  6   | `word_count`            |   4.34%    | **Content Depth:** Thin content is more easily displaced by comprehensive competitors.                                          |
|  7   | `char_count`            |   4.19%    | Textual granularity.                                                                                                            |
|  8   | `scroll_rate`           |   3.08%    | **On-Page Engagement:** Sub-baseline scroll depth signals weak reader satisfaction.                                             |
|  9   | `ctr`                   |   2.82%    | Snippet click capture rate.                                                                                                     |
|  10  | `clicks_90d`            |   2.82%    | Historical traffic baseline.                                                                                                    |

### Surprises and Negative Results

- **Staleness is not monotonically bad:** While intuition suggests the oldest pages always decay fastest, empirical importance of `days_since_last_update` was only 2.41%. Pages surviving $>180$ days without updates often represent stable evergreen assets or client batch logging gaps.
- **AI referral traffic was non-predictive:** `ai_traffic_pct` and `ai_sessions_90d` contributed $<0.5\%$ importance due to extreme sparsity (over 85% of pages had zero recorded AI referral sessions).

---

## 7. Action Recommendations & Editorial Playbook

### Ranked Content Action Playbook

The model's probability scores map directly into an operational 4-tier triage matrix for editorial workflows:

| Priority Tier                         |  Score Range  | Operational Action                                                                                                      | Reason Code                      |    Weekly Quota     |
| :------------------------------------ | :-----------: | :---------------------------------------------------------------------------------------------------------------------- | :------------------------------- | :-----------------: |
| **Tier 1: Immediate Triage**          |  $\ge 0.75$   | **Deep Content Refresh:** Update facts, add new sections, refresh dates, expand word count.                             | `high_demand_striking_decay`     |    Top 20 pages     |
| **Tier 2: Metadata Optimization**     | $0.60 - 0.74$ | **CTR / Snippet Rewrite:** Rewrite `<title>` and `<meta description>` to lift click capture without altering body text. | `page1_low_ctr_vulnerability`    |    Next 30 pages    |
| **Tier 3: Strategic Pruning / Merge** | $0.45 - 0.59$ | **Consolidation / Prune:** Merge thin, cannibalizing URLs or 301 redirect into flagship hub pages.                      | `thin_low_engagement_drag`       |   Bi-weekly batch   |
| **Tier 4: Passive Monitoring**        |   $< 0.45$    | **Monitor Only:** No editorial intervention required; preserve budget.                                                  | `stable_evergreen_or_low_volume` | Remaining inventory |

### Explicit Confidence & Operational Limitations

- **Decision-Support Framing:** The model ranks pages exhibiting statistical indicators of search decay. It **does not prove** that refreshing a page will guarantee ranking recovery.
- **Exogenous Search Shocks:** The model cannot anticipate external search engine redesigns (e.g., Google introducing AI Overviews directly above organic results).
- **Domain Authority Shifts:** Sitewide technical penalties or domain migrations will affect page traffic independently of page-level content freshness.

---

## 8. Reproducibility & Audit Trail

### Environment Specifications

- **Operating System:** Windows 10/11 x64 / Linux x64
- **Python Version:** 3.13.9 (compatible with Python $\ge 3.10$)
- **Key Libraries:** `pandas == 2.3.3`, `scikit-learn == 1.7.2`, `numpy == 2.2.6`, `nbconvert == 7.16.6`
- **Random Seeds:** `RANDOM_STATE = 42` fixed across all splitting, sampling, and model initialization.

### Re-Run Instructions from Clean Clone

To reproduce all numbers, comparison tables, and figures from a fresh clone:

```bash
# 1. Clone repository and install dependencies
git clone https://github.com/umerkang66/flyrank-ml-internship.git
cd flyrank-ml-internship
pip install -r requirements.txt

# 2. Re-run baseline and modeling notebooks top-to-bottom
python -m jupyter nbconvert --to notebook --execute --inplace work/notebooks/w04_baseline_score.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace work/notebooks/w05_model.ipynb
```

### Committed Audit Receipts

All numbers cited in this report trace back directly to committed machine receipts:

- Baseline metrics: [`work/outputs/baseline_metrics.json`](file:///D:/Workspace/internship_2026/work/outputs/baseline_metrics.json)
- Baseline ranked queue: [`work/outputs/baseline_action_score.csv`](file:///D:/Workspace/internship_2026/work/outputs/baseline_action_score.csv)
- Model comparison metrics: [`work/outputs/model_results.json`](file:///D:/Workspace/internship_2026/work/outputs/model_results.json)
- Full test predictions: `work/outputs/model_predictions.csv` (generated locally during run)

---

## 9. Acknowledgments & Data Credit

Built on the **[FlyRank ML Internship dataset](https://flyrank.ai)**. We gratefully acknowledge FlyRank for providing access to pseudonymized, production-grade search intelligence and performance telemetry across multi-client digital portfolios.
