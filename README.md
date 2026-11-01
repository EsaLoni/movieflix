# MovieFlix — Streamlit starter

A dark neon-purple movie discovery app with TMDB search, actor/director filmographies, movie details, cast, director, trailer links, similar movies, and India watch-provider links. It also includes an optional local CSV catalogue view.

## 1. Install

Open Command Prompt in the project folder and run:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2. Configure TMDB securely

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
2. Replace the placeholder with your **newly rotated** TMDB key.
3. Do not commit `secrets.toml` or paste keys into source code.

## 3. Run

```bash
streamlit run app.py
```

## 4. Use the local CSV

Put `IMDB_Data.csv` in the same folder as `app.py`, or set the `MOVIE_CSV_PATH` environment variable to the full path of your CSV. The Local catalogue page checks common title/overview/genre/cast/director column names and reports if it cannot find usable titles.

## Notes about the notebook

The notebook's current FAISS setup embeds all 760,803 rows on every fresh run unless the generated artifacts are loaded. Its flat index needs roughly 1.1 GiB just for 760,803 × 384 float32 vectors, before the model, pandas data, and other runtime memory. That is not a good fit for a small Streamlit Cloud deployment. This starter uses TMDB for the live discovery/details pages and keeps the CSV catalogue as an optional local feature. We should validate the CSV's title column and then decide whether to connect a smaller/precomputed recommendation index.
