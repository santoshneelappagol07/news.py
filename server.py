"""
server.py — Flask application for NIFTY Market Analysis Dashboard.
Serves the dashboard UI and provides the /api/analyze endpoint.
Includes Intraday predictions alongside BTST.
"""

import json
import logging
import traceback
from flask import Flask, render_template, jsonify

from scraper import scrape_all_news
from analyzer import analyze_news
from intraday_analyzer import generate_intraday_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Trigger full news scraping + sentiment analysis
    + intraday prediction. Returns the complete prediction JSON.
    """
    try:
        logger.info("━━━ Starting market analysis ━━━")

        # Phase 1: Scrape news
        logger.info("Phase 1: Scraping news from all sources...")
        news_items = scrape_all_news()
        logger.info(f"Scraped {len(news_items)} news items.")

        # Phase 2: Analyze sentiment (BTST prediction)
        logger.info("Phase 2: Running sentiment analysis (BTST)...")
        result = analyze_news(news_items)
        logger.info(
            f"BTST Analysis — Prediction: {result['prediction']}, "
            f"Confidence: {result['confidence']}%"
        )

        # Phase 3: Generate intraday prediction
        logger.info("Phase 3: Generating intraday prediction...")
        intraday = generate_intraday_prediction(
            news_sentiment=result["news_sentiment"],
            gap_prediction=result["prediction"],
            event_risk=result["event_risk"],
            scores=result["scores"],
            bullish_factors=result["bullish_factors"],
            bearish_factors=result["bearish_factors"],
            sector_summary=result["sector_summary"],
        )
        logger.info(
            f"Intraday — Bias: {intraday['intraday_bias']['bias']}, "
            f"Pattern: {intraday['intraday_pattern']['pattern']}, "
            f"Volatility: {intraday['volatility']['level']}"
        )

        # Merge everything into the result
        result["intraday"] = intraday

        return jsonify({"status": "success", "data": result})

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Analysis failed: {str(e)}",
        }), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "NIFTY Market Analyzer (BTST + Intraday)"})


if __name__ == "__main__":
    print("\n" + "━" * 60)
    print("  🚀 NIFTY Market Analysis Dashboard")
    print("  📊 BTST + Intraday Intelligence")
    print("  🌐 Open: http://localhost:5000")
    print("━" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
