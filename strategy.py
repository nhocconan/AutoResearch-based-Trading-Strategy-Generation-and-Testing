#!/usr/bin/env python3
"""
12h_MACD_RSI_Confluence_1dTrend
Hypothesis: On 12h timeframe, MACD bullish/bearish cross combined with RSI extremes 
(oversold/overbought) generates high-probability signals when aligned with 1d EMA50 trend 
and volume > 1.3x 20-period average. Uses 1d Bollinger Band width < 60th percentile 
to avoid choppy regimes. Targets 15-30 trades/year (60-120 total over 4 years) 
with low turnover. Works in bull via MACD momentum and bear via RSI mean-reversion 
with trend filter.
"""

name = "12h_MACD_RSI_Confluence_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 1d Bollinger Band width (20, 2) for squeeze filter
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma20_1d + 2 * std20_1d
    lower_bb_1d = sma20_1d - 2 * std20_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma20_1d
    # Percentile rank of bb_width over lookback
    bb_width_rank = pd.Series(bb_width_1d).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    bb_width_rank_aligned = align_htf_to_ltf(prices, df_1d, bb_width_rank)

    # MACD (12,26,9)
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line

    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume confirmation: 1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Get aligned values for current 12h bar
        ema50 = ema50_1d_aligned[i]
        bb_rank = bb_width_rank_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(bb_rank) or 
            np.isnan(macd_line[i]) or np.isnan(signal_line[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Squeeze filter: only trade when BB width is in lower 60% (avoid chop)
        if bb_rank > 0.6:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        vol_avg_val = vol_avg_20[i]

        if position == 0:
            # LONG: MACD bullish cross + RSI oversold + price above EMA50 + volume surge
            if (macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1] and
                rsi[i] < 35 and
                close[i] > ema50 and
                volume[i] > vol_avg_val * 1.3):
                signals[i] = 0.25
                position = 1
            # SHORT: MACD bearish cross + RSI overbought + price below EMA50 + volume surge
            elif (macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1] and
                  rsi[i] > 65 and
                  close[i] < ema50 and
                  volume[i] > vol_avg_val * 1.3):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: MACD bearish cross or RSI overbought or price below EMA50
            if (macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]) or \
               (rsi[i] > 70) or \
               (close[i] < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: MACD bullish cross or RSI oversold or price above EMA50
            if (macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]) or \
               (rsi[i] < 30) or \
               (close[i] > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals