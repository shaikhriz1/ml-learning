# ML Project Starter

A minimal, job-ready Machine Learning project structure to kickstart your portfolio.

## What's inside
- `src/train.py` – Scikit-learn training script
- `notebooks/` – keep your EDA and experiments
- `data/` – **store small sample data only** (avoid pushing big/private data)
- `models/` – trained models (keep small or ignore with .gitignore)
- `app/streamlit_app.py` – simple demo app
- `.github/workflows/ci.yml` – basic CI to check the project installs
- `.gitignore` – keeps junk and large files out of git

## Quickstart
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Put a small CSV into `data/raw/dataset.csv` (or edit `src/train.py` to your path).
4. Train a model:
   ```bash
   python src/train.py
   ```
5. Run the demo app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Notes
- Do **not** commit sensitive data or credentials.
- Prefer linking to data sources (Kaggle, etc.) instead of pushing big files.
- If you must track large files, consider Git LFS or DVC.
