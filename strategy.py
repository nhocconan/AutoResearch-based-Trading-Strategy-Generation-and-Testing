#!/usr/bin/env python3
"""
12h_WeeklyPivot_Squeeze_Breakout_Trend
Hypothesis: On 12h timeframe, weekly pivot levels act as strong support/resistance. 
Breakouts above R1 or below S1 with trend confirmation (1d EMA50) and low volatility 
(Bollinger Band width < 30th percentile) generate signals. Uses volatility filter to 
avoid chop and focus on breakouts from consolidation. Works in bull via momentum 
breaks and bear via mean-reversion at extremes with trend filter. Targets 15-25 
trades/year (60-100 total over 4 years) to minimize fee drag.
"""

name = "12h_WeeklyPivot_Squeeze_Breakout_Trend"
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

    # Calculate 1d Bollinger Band width (20, 2) for volatility filter
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

    # Calculate weekly pivot levels from previous week OHLC
    # Get weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    pivot = (high_1w + low_1w + close_1w) / 3
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Align weekly data to 12h timeframe (weekly data needs no shift as it's already based on previous week)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)

    # Volume confirmation: 1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Get aligned values for current 12h bar
        ema50 = ema50_1d_aligned[i]
        bb_rank = bb_width_rank_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(bb_rank) or 
            np.isnan(r1_level) or np.isnan(s1_level) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volatility filter: only trade when BB width is in lower 30% (low volatility/squeeze)
        if bb_rank > 0.3:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + price above EMA50 + volume surge
            if (close[i] > r1_level and 
                close[i] > ema50 and 
                volume[i] > vol_avg_val * 1.3):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + price below EMA50 + volume surge
            elif (close[i] < s1_level and 
                  close[i] < ema50 and 
                  volume[i] > vol_avg_val * 1.3):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or price below EMA50
            if (close[i] < s1_level or close[i] < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or price above EMA50
            if (close[i] > r1_level or close[i] > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals