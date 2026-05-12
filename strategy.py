#!/usr/bin/env python3
"""
4h_MultiTF_Camarilla_Breakout_TrendFilter
Hypothesis: On 4h timeframe, Camarilla R3/S3 levels from 1d act as strong support/resistance.
Breaks above R3 with 1d EMA34 uptrend and volume > 1.5x 20-period average generate long signals;
breaks below S3 with 1d EMA34 downtrend and volume surge generate shorts.
Uses 1d Bollinger Band width < 50th percentile to filter choppy regimes.
Targets 20-50 trades/year (80-200 total over 4 years) with low turnover to minimize fee drift.
Works in bull via momentum breaks and bear via mean-reversion at extremes with trend filter.
"""

name = "4h_MultiTF_Camarilla_Breakout_TrendFilter"
timeframe = "4h"
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

    # Calculate 1d Camarilla levels from previous 1d OHLC
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # We need previous day's HLC, so shift by 1
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First value will be invalid, handled by alignment
    camarilla_mult = 1.1 / 4
    r3 = prev_close + (prev_high - prev_low) * camarilla_mult
    s3 = prev_close - (prev_high - prev_low) * camarilla_mult
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Get aligned values for current 4h bar
        ema34 = ema34_1d_aligned[i]
        bb_rank = bb_width_rank_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema34) or np.isnan(bb_rank) or 
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
            # LONG: Price breaks above R3 + price above EMA34 + volume surge
            if (close[i] > r3_level and 
                close[i] > ema34 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + price below EMA34 + volume surge
            elif (close[i] < s3_level and 
                  close[i] < ema34 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or price below EMA34
            if (close[i] < s3_level or close[i] < ema34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or price above EMA34
            if (close[i] > r3_level or close[i] > ema34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals