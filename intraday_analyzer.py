"""
intraday_analyzer.py — Intraday market movement prediction engine.

Combines news sentiment, global cues, and time-of-day factors to
generate intraday trading predictions for NIFTY 50.

Rules & Conditions (based on professional trading heuristics):
─────────────────────────────────────────────────────────────────
1. NEWS SENTIMENT IMPACT ON INTRADAY:
   - Strong bullish news → Trending day up
   - Strong bearish news → Trending day down
   - Mixed signals → Range-bound / choppy day
   - Event risk HIGH → Volatile / whipsaw expected

2. TIME-OF-DAY FACTORS:
   - Pre-market (before 9:15) → Gap prediction is key
   - First 30 min (9:15-9:45) → High volatility, gap fill or gap extension
   - Mid-morning (9:45-11:30) → Trend formation phase
   - Lunch (11:30-1:30) → Low volume, choppy
   - Afternoon (1:30-3:00) → Institutional activity, closing trend
   - Last 15 min (3:00-3:30) → Short covering / profit booking

3. INTRADAY PATTERNS:
   - Gap Up + Bullish → Buy on dips (support at gap)
   - Gap Up + Bearish → Possible gap fill, sell on rise
   - Gap Down + Bearish → Sell on rise (resistance at gap)
   - Gap Down + Bullish → Possible gap fill, buy on dips
   - Flat + Strong sentiment → Trending move expected
   - Flat + Mixed → Range-bound, sell CE + PE (option selling)

4. VOLATILITY ESTIMATION:
   - Event risk + strong sentiment → High volatility
   - Low activity + no events → Low volatility (option selling opportunity)

5. SUPPORT & RESISTANCE LOGIC (estimated from sentiment):
   - Strong bullish: Support well-defined, resistance breakout possible
   - Strong bearish: Resistance strong, support breakdown possible
   - Mixed: Both levels respected, range-bound
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# IST timezone offset
IST = timezone(timedelta(hours=5, minutes=30))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Time-of-Day Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_market_phase() -> dict:
    """Determine the current market phase based on IST time."""
    now = datetime.now(IST)
    hour = now.hour
    minute = now.minute
    time_val = hour * 100 + minute

    if time_val < 900:
        return {
            "phase": "PRE-MARKET",
            "description": "Markets haven't opened yet. Focus on global cues and overnight news.",
            "volatility_factor": 1.0,
            "icon": "🌅",
        }
    elif time_val < 915:
        return {
            "phase": "PRE-OPEN",
            "description": "Pre-open session active. Gap prediction is most relevant.",
            "volatility_factor": 1.2,
            "icon": "⏰",
        }
    elif time_val < 945:
        return {
            "phase": "OPENING HALF-HOUR",
            "description": "High volatility opening phase. Gap fill or extension pattern determines the day.",
            "volatility_factor": 1.5,
            "icon": "🔥",
        }
    elif time_val < 1130:
        return {
            "phase": "TREND FORMATION",
            "description": "The day's trend is being established. Watch for breakout/breakdown from opening range.",
            "volatility_factor": 1.2,
            "icon": "📈",
        }
    elif time_val < 1330:
        return {
            "phase": "LUNCH SESSION",
            "description": "Low volume lunch period. Markets tend to be range-bound and choppy.",
            "volatility_factor": 0.7,
            "icon": "🍽️",
        }
    elif time_val < 1500:
        return {
            "phase": "AFTERNOON TREND",
            "description": "Institutional activity picks up. Closing trend starts forming.",
            "volatility_factor": 1.3,
            "icon": "📊",
        }
    elif time_val < 1530:
        return {
            "phase": "CLOSING SESSION",
            "description": "Last 30 minutes — short covering, profit booking, and closing moves.",
            "volatility_factor": 1.4,
            "icon": "🏁",
        }
    else:
        return {
            "phase": "AFTER MARKET",
            "description": "Markets are closed. Focus shifts to next-day prediction (BTST).",
            "volatility_factor": 0.5,
            "icon": "🌙",
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Intraday Pattern Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _detect_intraday_pattern(
    gap_prediction: str,
    news_sentiment: str,
    sentiment_score: float,
    event_risk: str,
) -> dict:
    """
    Detect the expected intraday pattern based on gap prediction,
    sentiment, and news scores.
    """
    # Pattern matrix
    if gap_prediction == "GAP UP":
        if news_sentiment == "BULLISH" and sentiment_score > 5:
            return {
                "pattern": "TRENDING UP",
                "description": "Strong gap up with bullish sentiment. Expect a trending up day with higher highs.",
                "strategy": "Buy on dips. Trail stop-loss at opening price. Target: 1-1.5% above open.",
                "option_strategy": "Buy CE at opening dip. Sell PE for premium collection.",
                "risk_level": "MODERATE",
            }
        elif news_sentiment == "BEARISH":
            return {
                "pattern": "GAP FILL",
                "description": "Gap up against bearish pressure. High probability of gap fill during the day.",
                "strategy": "Sell on rise near resistance. Target: Previous day's close. Stop-loss: Day's high.",
                "option_strategy": "Buy PE or sell CE at higher strikes. Gap fill probability is high.",
                "risk_level": "HIGH",
            }
        else:
            return {
                "pattern": "RANGE + UPSIDE BIAS",
                "description": "Gap up with mixed signals. Likely to consolidate with mild upside bias.",
                "strategy": "Buy near day's low, sell near day's high. Keep tight stop-losses.",
                "option_strategy": "Short strangle/straddle if premium is good. Buy CE for directional play.",
                "risk_level": "MODERATE",
            }

    elif gap_prediction == "GAP DOWN":
        if news_sentiment == "BEARISH" and sentiment_score < -5:
            return {
                "pattern": "TRENDING DOWN",
                "description": "Strong gap down with bearish sentiment. Expect a trending down day with lower lows.",
                "strategy": "Sell on rise. Trail stop-loss at opening price. Target: 1-1.5% below open.",
                "option_strategy": "Buy PE at opening bounce. Sell CE for premium collection.",
                "risk_level": "MODERATE",
            }
        elif news_sentiment == "BULLISH":
            return {
                "pattern": "GAP FILL UP",
                "description": "Gap down against bullish support. High probability of gap fill recovery during the day.",
                "strategy": "Buy on dips near support. Target: Previous day's close. Stop-loss: Day's low.",
                "option_strategy": "Buy CE or sell PE at lower strikes. Gap fill recovery expected.",
                "risk_level": "HIGH",
            }
        else:
            return {
                "pattern": "RANGE + DOWNSIDE BIAS",
                "description": "Gap down with mixed signals. Likely to consolidate with mild downside bias.",
                "strategy": "Sell near day's high, cover near day's low. Keep tight stop-losses.",
                "option_strategy": "Short strangle if premium is fat. Buy PE for directional play.",
                "risk_level": "MODERATE",
            }

    else:  # FLAT
        if event_risk == "HIGH":
            return {
                "pattern": "VOLATILE WHIPSAW",
                "description": "Flat opening with high event risk. Expect volatile whipsaw moves in both directions.",
                "strategy": "Wait for directional clarity. Avoid early trades. Trade after event outcome is clear.",
                "option_strategy": "Buy straddle/strangle for volatility play. Avoid directional bets.",
                "risk_level": "VERY HIGH",
            }
        elif abs(sentiment_score) > 10:
            direction = "UPWARD" if sentiment_score > 0 else "DOWNWARD"
            return {
                "pattern": f"BREAKOUT {direction}",
                "description": f"Flat opening but strong news sentiment suggests {direction.lower()} breakout during the day.",
                "strategy": f"Wait for breakout confirmation past opening range. Trade in direction of sentiment.",
                "option_strategy": f"Buy {'CE' if sentiment_score > 0 else 'PE'} after opening range breakout for trending move.",
                "risk_level": "MODERATE",
            }
        elif news_sentiment == "MIXED":
            return {
                "pattern": "RANGE-BOUND",
                "description": "Flat opening with mixed sentiment. Expect a range-bound day with no clear trend.",
                "strategy": "Sell at resistance, buy at support. Trade the range with tight risk management.",
                "option_strategy": "Sell straddle/strangle. Premium collection day. Time decay is your friend.",
                "risk_level": "LOW",
            }
        else:
            sentiment_dir = "bullish" if news_sentiment == "BULLISH" else "bearish"
            return {
                "pattern": f"SLOW {sentiment_dir.upper()} DRIFT",
                "description": f"Flat opening with {sentiment_dir} tilt. Expect a slow directional drift throughout the day.",
                "strategy": f"Trade with the sentiment. {'Buy dips' if sentiment_dir == 'bullish' else 'Sell rises'} with patience.",
                "option_strategy": f"Buy {'CE' if sentiment_dir == 'bullish' else 'PE'} ITM for directional play.",
                "risk_level": "LOW",
            }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Volatility Score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _estimate_volatility(
    event_risk: str,
    news_count: int,
    sentiment_score_diff: float,
) -> dict:
    """Estimate intraday volatility level."""
    vol_score = 0

    # Base volatility from news volume
    if news_count > 80:
        vol_score += 2
    elif news_count > 40:
        vol_score += 1

    # Event risk
    if event_risk == "HIGH":
        vol_score += 3
    elif event_risk == "MEDIUM":
        vol_score += 1

    # News sentiment divergence
    if abs(sentiment_score_diff) > 20:
        vol_score += 3
    elif abs(sentiment_score_diff) > 10:
        vol_score += 2
    elif abs(sentiment_score_diff) > 5:
        vol_score += 1

    # Apply market phase factor
    phase = _get_market_phase()
    vol_score = int(vol_score * phase["volatility_factor"])

    # Classify
    if vol_score >= 8:
        level = "EXTREME"
        expected_range = "200-400 pts"
        nifty_range_pct = "1.0-2.0%"
    elif vol_score >= 5:
        level = "HIGH"
        expected_range = "100-200 pts"
        nifty_range_pct = "0.5-1.0%"
    elif vol_score >= 3:
        level = "MODERATE"
        expected_range = "50-100 pts"
        nifty_range_pct = "0.25-0.5%"
    else:
        level = "LOW"
        expected_range = "30-50 pts"
        nifty_range_pct = "0.1-0.25%"

    return {
        "level": level,
        "score": vol_score,
        "expected_range": expected_range,
        "nifty_range_pct": nifty_range_pct,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Intraday Bias + Strategy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _generate_intraday_bias(
    news_score_diff: float,
    event_risk: str,
    volatility_level: str,
    gap_prediction: str,
) -> dict:
    """Generate the intraday trading bias and actionable levels."""
    total_score = news_score_diff / 5  # Normalize news score

    if event_risk == "HIGH" or volatility_level == "EXTREME":
        bias = "VOLATILE — AVOID"
        icon = "⚡"
        confidence = 25
    elif total_score > 5:
        bias = "STRONGLY BULLISH"
        icon = "🟢🟢"
        confidence = min(80, 50 + int(total_score * 3))
    elif total_score > 2:
        bias = "BULLISH"
        icon = "🟢"
        confidence = min(70, 40 + int(total_score * 4))
    elif total_score > 0:
        bias = "MILDLY BULLISH"
        icon = "🟡🟢"
        confidence = min(55, 35 + int(total_score * 5))
    elif total_score > -2:
        bias = "MILDLY BEARISH"
        icon = "🟡🔴"
        confidence = min(55, 35 + int(abs(total_score) * 5))
    elif total_score > -5:
        bias = "BEARISH"
        icon = "🔴"
        confidence = min(70, 40 + int(abs(total_score) * 4))
    else:
        bias = "STRONGLY BEARISH"
        icon = "🔴🔴"
        confidence = min(80, 50 + int(abs(total_score) * 3))

    # Build strategy
    strategies = []
    if "BULLISH" in bias:
        strategies.append("Buy CE (Call Options) on morning dips")
        strategies.append("Sell PE (Put Options) for premium collection")
        if gap_prediction == "GAP UP":
            strategies.append("Trail SL at opening price — let winners run")
        elif gap_prediction == "GAP DOWN":
            strategies.append("Buy on gap down — anticipate recovery")
    elif "BEARISH" in bias:
        strategies.append("Buy PE (Put Options) on morning bounce")
        strategies.append("Sell CE (Call Options) for premium collection")
        if gap_prediction == "GAP DOWN":
            strategies.append("Trail SL at opening price — let winners run")
        elif gap_prediction == "GAP UP":
            strategies.append("Sell on gap up — anticipate reversal")
    else:
        strategies.append("Wait for directional clarity before taking trades")
        strategies.append("If forced to trade, sell straddle/strangle for premium")
        strategies.append("Keep position sizes very small")

    return {
        "bias": bias,
        "icon": icon,
        "confidence": confidence,
        "strategies": strategies,
        "total_score": round(total_score, 1),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def generate_intraday_prediction(
    news_sentiment: str,
    gap_prediction: str,
    event_risk: str,
    scores: dict,
    bullish_factors: list[str],
    bearish_factors: list[str],
    sector_summary: list[dict],
) -> dict[str, Any]:
    """
    Generate complete intraday prediction using available data.

    Parameters
    ----------
    news_sentiment : str
        Overall news sentiment (BULLISH/BEARISH/MIXED).
    gap_prediction : str
        BTST gap prediction (GAP UP/GAP DOWN/FLAT).
    event_risk : str
        Event risk level (HIGH/MEDIUM/LOW).
    scores : dict
        News sentiment scores from analyzer.
    bullish_factors, bearish_factors : list[str]
        Key factors from the news analyzer.
    sector_summary : list[dict]
        Sector-wise sentiment breakdown.

    Returns
    -------
    dict
        Complete intraday prediction with bias, pattern, strategy, and more.
    """
    logger.info("Generating intraday prediction...")

    # ── Market phase ──
    market_phase = _get_market_phase()

    # ── Sentiment score ──
    score_diff = scores.get("net_score", 0)
    news_count = scores.get("total_bullish", 0) + scores.get("total_bearish", 0)

    # ── Intraday pattern ──
    intraday_pattern = _detect_intraday_pattern(
        gap_prediction, news_sentiment, score_diff, event_risk,
    )

    # ── Volatility estimation ──
    volatility = _estimate_volatility(
        event_risk, int(news_count), score_diff,
    )

    # ── Intraday bias ──
    intraday_bias = _generate_intraday_bias(
        score_diff, event_risk, volatility["level"], gap_prediction,
    )

    # ── Key intraday drivers ──
    intraday_drivers = []

    # Add news-based drivers
    if news_sentiment == "BULLISH":
        intraday_drivers.append("News sentiment strongly bullish — supports upside intraday")
    elif news_sentiment == "BEARISH":
        intraday_drivers.append("News sentiment strongly bearish — supports downside intraday")
    else:
        intraday_drivers.append("News sentiment mixed — choppy intraday expected")

    if event_risk == "HIGH":
        intraday_drivers.append("⚠️ HIGH EVENT RISK — Market can whipsaw violently. Reduce position sizes!")

    # Add score context
    if abs(score_diff) > 15:
        direction = "bullish" if score_diff > 0 else "bearish"
        intraday_drivers.append(f"Strong {direction} sentiment score ({score_diff:+.1f}) — trending day likely")
    elif abs(score_diff) > 5:
        direction = "bullish" if score_diff > 0 else "bearish"
        intraday_drivers.append(f"Moderate {direction} sentiment score ({score_diff:+.1f})")

    # Add sector highlights relevant to intraday
    bullish_sectors = [s["sector"] for s in sector_summary if s.get("sentiment") == "BULLISH"]
    bearish_sectors = [s["sector"] for s in sector_summary if s.get("sentiment") == "BEARISH"]

    if bullish_sectors:
        intraday_drivers.append(f"Bullish sectors for intraday: {', '.join(bullish_sectors[:3])}")
    if bearish_sectors:
        intraday_drivers.append(f"Bearish sectors for intraday: {', '.join(bearish_sectors[:3])}")

    # ── Intraday summary ──
    summary = _generate_intraday_summary(
        intraday_bias, intraday_pattern, market_phase,
        volatility, event_risk, score_diff,
    )

    return {
        "intraday_bias": intraday_bias,
        "intraday_pattern": intraday_pattern,
        "market_phase": market_phase,
        "volatility": volatility,
        "intraday_drivers": intraday_drivers[:8],
        "intraday_summary": summary,
    }


def _generate_intraday_summary(
    bias: dict,
    pattern: dict,
    phase: dict,
    volatility: dict,
    event_risk: str,
    score_diff: float,
) -> str:
    """Generate a human-readable intraday summary."""
    parts = []

    # Sentiment summary
    if score_diff > 5:
        parts.append(f"News sentiment is bullish with a net positive score of +{score_diff:.1f}.")
    elif score_diff < -5:
        parts.append(f"News sentiment is bearish with a net negative score of {score_diff:.1f}.")
    else:
        parts.append("News sentiment is mixed with no clear directional bias.")

    # Bias
    parts.append(f"Intraday bias is {bias['bias']} with {bias['confidence']}% confidence.")

    # Pattern
    parts.append(f"Expected pattern: {pattern['pattern']} — {pattern['description']}")

    # Volatility
    parts.append(
        f"Expected volatility: {volatility['level']} "
        f"(NIFTY range: {volatility['expected_range']}, ~{volatility['nifty_range_pct']})."
    )

    # Market phase
    parts.append(f"Current market phase: {phase['phase']} — {phase['description']}")

    # Event risk warning
    if event_risk == "HIGH":
        parts.append(
            "⚠️ HIGH EVENT RISK: Major event imminent. Markets may swing wildly. "
            "Intraday traders should use strict stop-losses and reduced position sizes."
        )

    return " ".join(parts)
