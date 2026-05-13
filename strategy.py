#/usr/bin/env python3
# 1d_RSIOversoldBullishEngulfing_WeeklyTrend
# Hypothesis: Buy when RSI(14) < 30 and a bullish engulfing candle forms on the daily chart,
# filtered by weekly trend (price > weekly EMA50). Exit when RSI > 70 or trend flips.
# Designed to capture mean reversion bounces in uptrends and avoid downtrends.
# Works in bull markets via trend-following bias and in bear markets via strict oversold entry.
# Target: 10-25 trades/year per symbol to minimize fee drag.

name = "1d_RSIOversoldBullishEngulfing_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')

    # RSI(14) on daily
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Bullish engulfing: current candle fully engulfs previous bearish candle
    bullish_engulfing = (close > open_) & (open_ < close) & (close > open_[1]) & (open_ < close[1]) & (close[1] < open_[1])

    # Weekly EMA50 trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long

    for i in range(1, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(bullish_engulfing[i])):
            if position == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI oversold + bullish engulfing + weekly uptrend
            if (rsi[i] < 30 and 
                bullish_engulfing[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT: RSI overbought OR trend turns down
            if rsi[i] > 70 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25

    return signals