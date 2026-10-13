# Final presentation plan (50 minutes, 3 presenters + timekeeper)

The order follows the rubric and the required agenda in the guidelines. Assign one member as **timekeeper**, and keep every slide readable from the back row: at most 6 lines, with large charts.

| # | Slide | Content (use the figures in `docs/images` and `outputs/*/figures`) | Speaker | Min |
|---|---|---|---|---|
| 1 | Title & team | BusinessLaunch AI, team names, roles | all | 1 |
| 2 | Agenda | what we will cover | M3 | 1 |
| 3 | Problem & motivation | the question; 61% of Philly restaurants/cafés still open; who the user is | M3 | 3 |
| 4 | Dataset | Yelp Open Dataset: business 150k → Philadelphia 14,570 → 6,174 restaurants/cafés; 704k reviews; 1.29M check-ins | M1 | 2 |
| 5 | Preprocessing pipeline | city/state filter, coordinate checks, duplicates, attribute/hours parsing, review cleaning with negation kept | M1 + M2 | 3 |
| 6 | System architecture | `docs/ARCHITECTURE.md` diagram: one ML notebook (3 parts) → outputs → shared package → app | M3 | 2 |
| 7 | M1 · EDA | category, rating and location charts; postal-code concentration | M1 | 3 |
| 8 | M1 · Feature engineering | haversine BallTree competitor counts, check-in activity, business age, attributes | M1 | 3 |
| 9 | M1 · Models & evaluation | spatial split map; LR / RF / GB / HistGB; MAE, RMSE, R²; **R² 0.06 → 0.61** after feature engineering; observed-vs-estimated plot | M1 | 4 |
| 10 | M2 · Review EDA | star distribution, review length, frequent words | M2 | 2 |
| 11 | M2 · Sentiment models | TF-IDF, NB vs LR, GridSearch with GroupKFold, macro F1 **0.944**, confusion matrix, error analysis (mixed reviews) | M2 | 4 |
| 12 | M2 · Complaint topics | NMF, choosing k by topic diversity, 6 named themes with examples | M2 | 3 |
| 13 | M3 · Market clustering | 13 neighbourhood features; K-Means vs Ward vs GMM; silhouette/DB/CH; stability; cluster map + profile heatmap | M3 | 4 |
| 14 | M3 · Financial simulator | break-even, scenarios, Monte Carlo histogram, sensitivity tornado | M3 | 3 |
| 15 | M3 · Recommendation logic | opportunity / risk percentile scores, verdict rules, alternative locations | M3 | 2 |
| 16 | **Live demo** | Center City vs East Passyunk vs Northeast: Home → Reviews → Clusters → Finance → Launch Report, then download the report | M3 (M1/M2 narrate their pages) | 7 |
| 17 | Challenges & solutions | weak location-only model (fixed with features), leakage risks (spatial/business splits), 5 GB review file (chunked streaming), merging 3 members (shared package) | all | 2 |
| 18 | Limitations & future work | README lists; licence data validation, census/rent data, transformers | all | 1 |
| 19 | Q&A | — | all | ~0 (remaining time) |

**Demo safety checklist:** start the app before the session (`streamlit run app/Home.py`); preload the pages once so caches are warm; have screenshots of each page ready in case the projector or Wi-Fi fails (map tiles need internet, everything else runs offline).

**Likely questions to prepare:**
* Why is review count a valid target? (It's a popularity proxy, not revenue, and it's corrected for age via years on Yelp.)
* Why macro F1 instead of accuracy? (Classes are 79/21 imbalanced.)
* Why is the silhouette only 0.23? (Areas blend into each other. The clusters are stable, ARI 0.87, and best among the three methods.)
* How did you avoid leakage? (Spatial and business-grouped splits, plus assertions.)
* Where do the financial numbers come from? (User assumptions; the data informs only the defaults.)
