#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Volume_Trend
Hypothesis: On 12h timeframe, Donchian(20) breakouts with volume > 1.5x 20-period average 
and 1d EMA34 trend alignment generate high-probability entries. The 1d Bollinger Band 
width < 50th percentile filters choppy regimes to avoid false breakouts. 
Exits when price retests the Donchian midpoint or violates the 1d EMA34 trend.
Designed for 12-37 trades/year (50-150 total over 4 years) with low turnover to minimize fee drag.
Works in bull via momentum breakouts and bear via mean-reversion at extremes with trend filter.
"""

name = "12h_Donchian_Breakout_Volume_Trend"
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

    # Calculate 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

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

    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Get aligned values for current 12h bar
        ema34 = ema34_1d_aligned[i]
        bb_rank = bb_width_rank_aligned[i]
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        dc_mid = donchian_mid[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema34) or np.isnan(bb_rank) or 
            np.isnan(dc_high) or np.isnan(dc_low) or 
            np.isnan(dc_mid) or np.isnan(vol_avg_val)):
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
            # LONG: Price breaks above Donchian high + price above EMA34 + volume surge
            if (close[i] > dc_high and 
                close[i] > ema34 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + price below EMA34 + volume surge
            elif (close[i] < dc_low and 
                  close[i] < ema34 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retests Donchian midpoint or violates EMA34 trend
            if (close[i] < dc_mid or close[i] < ema34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price retests Donchian midpoint or violates EMA34 trend
            if (close[i] > dc_mid or close[i] > ema34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals