#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_12hTrend_Volume
Hypothesis: On 6h timeframe, Camarilla R3/S3 levels from prior 12h act as strong support/resistance.
Breaks above R3 with 12h EMA50 uptrend and volume > 2x 20-period average generate long signals;
breaks below S3 with 12h EMA50 downtrend and volume > 2x average generate shorts.
Uses 12h Bollinger Band width < 50th percentile to filter choppy regimes.
Targets 12-37 trades/year (50-150 total over 4 years) with low turnover to minimize fee drag.
Works in bull markets via breakout momentum and in bear via mean reversion at extreme levels.
"""

name = "6h_Camarilla_R3S3_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Calculate 12h EMA50 for trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Calculate 12h Bollinger Band width (20, 2) for squeeze filter
    sma20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_bb_12h = sma20_12h + 2 * std20_12h
    lower_bb_12h = sma20_12h - 2 * std20_12h
    bb_width_12h = (upper_bb_12h - lower_bb_12h) / sma20_12h
    # Percentile rank of bb_width over lookback
    bb_width_rank = pd.Series(bb_width_12h).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    bb_width_rank_aligned = align_htf_to_ltf(prices, df_12h, bb_width_rank)

    # Calculate 12h Camarilla levels from previous 12h OHLC
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_close = np.roll(close_12h, 1)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    camarilla_mult = 1.1 / 4
    r3 = prev_close + (prev_high - prev_low) * camarilla_mult
    s3 = prev_close - (prev_high - prev_low) * camarilla_mult
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)

    # Volume confirmation: 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get aligned values for current 6h bar
        ema50 = ema50_12h_aligned[i]
        bb_rank = bb_width_rank_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(bb_rank) or 
            np.isnan(r3_level) or np.isnan(s3_level) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Squeeze filter: only trade when BB width is in lower 50% (contraction)
        if bb_rank > 0.5:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + price above EMA50 + volume surge
            if (close[i] > r3_level and 
                close[i] > ema50 and 
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + price below EMA50 + volume surge
            elif (close[i] < s3_level and 
                  close[i] < ema50 and 
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or price below EMA50
            if (close[i] < s3_level or close[i] < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or price above EMA50
            if (close[i] > r3_level or close[i] > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals