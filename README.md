# Accommodation Explorer

This Flask website explores Francistown accommodation listings and their estimated value.
At startup it trains a scaled Ridge regression on listing price, using review count,
review score, and distance from the city centre as predictors. Fair prices and the
over/under-price flags are based on leave-one-out predictions; the displayed error
band is the leave-one-out mean absolute error.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the website with `python app.py`.
4. Open `http://127.0.0.1:5000` in a browser.

The application reads listing data from `data/listings_scored.json`. The `.venv`
environment, Python cache files, local database files, and environment secrets are
excluded from Git.
