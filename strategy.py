#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Touch_1wTrend_VolumeFilter
Hypothesis: On 12h timeframe, price touching Camarilla R3/S3 levels from the previous week with alignment to 1w EMA trend and volume expansion (>1.5x 20-period average) captures institutional reversal points. Uses weekly Bollinger Band width (<40th percentile) to filter choppy conditions. Designed for low turnover (target 15-25 trades/year) to work in both bull and bear markets by fading extremes in ranging markets and catching reversals in trending markets.
"""

name = "12h_Camarilla_Pivot_Touch_1wTrend_VolumeFilter"
timeframe = "12h"
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

    # Get 1w data (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate 1w EMA40 for trend
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)

    # Calculate 1w Bollinger Band width (20, 2) for chop filter
    sma20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = sma20_1w + 2 * std20_1w
    lower_bb_1w = sma20_1w - 2 * std20_1w
    bb_width_1w = (upper_bb_1w - lower_bb_1w) / sma20_1w
    # Percentile rank of bb_width over lookback
    bb_width_rank = pd.Series(bb_width_1w).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    bb_width_rank_aligned = align_htf_to_ltf(prices, df_1w, bb_width_rank)

    # Calculate weekly Camarilla levels from previous 1w OHLC
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    camarilla_mult = 1.1 / 4
    r3 = prev_close + (prev_high - prev_low) * camarilla_mult
    s3 = prev_close - (prev_high - prev_low) * camarilla_mult
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get aligned values for current 12h bar
        ema40 = ema40_1w_aligned[i]
        bb_rank = bb_width_rank_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema40) or np.isnan(bb_rank) or 
            np.isnan(r3_level) or np.isnan(s3_level) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Chop filter: only trade when BB width is in lower 40% (contraction)
        if bb_rank > 0.4:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches or crosses above S3 + price below EMA40 (mean reversion in uptrend)
            if (close[i] <= s3_level and 
                close[i] < ema40 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches or crosses below R3 + price above EMA40 (mean reversion in downtrend)
            elif (close[i] >= r3_level and 
                  close[i] > ema40 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above EMA40 or touches R3 (take profit at median)
            if (close[i] >= ema40 or close[i] >= r3_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below EMA40 or touches S3 (take profit at median)
            if (close[i] <= ema40 or close[i] <= s3_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals