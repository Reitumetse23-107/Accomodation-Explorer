import json
from pathlib import Path

from flask import Flask, render_template

from pricing_model import train_pricing_model

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DATA = json.loads(
    (BASE_DIR / "data" / "listings_scored.json").read_text(encoding="utf-8")
)
EXPLORER_DATA, PRICING_MODEL = train_pricing_model(SOURCE_DATA)


@app.route("/")
def index():
    return render_template("index.html", explorer_data=EXPLORER_DATA)


if __name__ == "__main__":
    app.run(debug=True)
