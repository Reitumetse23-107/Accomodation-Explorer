import json
from pathlib import Path

from flask import Flask, render_template

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
EXPLORER_DATA = json.loads(
    (BASE_DIR / "data" / "listings_scored.json").read_text(encoding="utf-8")
)


@app.route("/")
def index():
    return render_template("index.html", explorer_data=EXPLORER_DATA)


if __name__ == "__main__":
    app.run(debug=True)
