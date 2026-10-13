# Data

The raw Yelp files are not committed (they are 100 MB – 5 GB). Download the **Yelp Open Dataset** from
[Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) or [yelp.com/dataset](https://www.yelp.com/dataset)
and put these files in this folder. The notebooks accept the plain `.json` or a `.json.zip` containing it.

| File | Size | Used by |
|---|---|---|
| `yelp_academic_dataset_business.json` (or `.json.zip`) | 119 MB | notebooks 01, 02, 03 |
| `yelp_academic_dataset_checkin.json` (or `.json.zip`) | 287 MB | notebooks 01, 03 |
| `yelp_academic_dataset_review.json` | 5.3 GB | notebook 02 |

The Streamlit app does **not** need these files: it only reads the exported results in `outputs/`.
