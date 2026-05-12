#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_WeeklyTrend
Hypothesis: On 6h timeframe, Camarilla R3/S3 levels from prior 1d act as strong breakout levels.
Trades only when both 1d EMA34 and weekly EMA34 agree on trend direction to avoid whipsaw.
Requires volume > 1.5x 20-period average for confirmation.
Targets 12-37 trades/year (50-150 total over 4 years) with low turnover to minimize fee drag.
Works in bull via momentum breaks and bear via trend-filtered breakouts.
"""

name = "6h_Camarilla_R3S3_Breakout_1dTrend_WeeklyTrend"
timeframe = "6h"
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

    # Get weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values

    # Calculate 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate weekly EMA34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Calculate 6h Camarilla levels from previous 1d OHLC
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
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
        # Get aligned values for current 6h bar
        ema34_1d_val = ema34_1d_aligned[i]
        ema34_1w_val = ema34_1w_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_val) or np.isnan(ema34_1w_val) or 
            np.isnan(r3_level) or np.isnan(s3_level) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: only trade when 1d and weekly EMA34 agree
        trend_agree = (close[i] > ema34_1d_val) == (close[i] > ema34_1w_val)

        if position == 0:
            # LONG: Price breaks above R3 + trend agreement + volume surge
            if (close[i] > r3_level and 
                trend_agree and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + trend agreement + volume surge
            elif (close[i] < s3_level and 
                  trend_agree and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or trend disagreement
            if (close[i] < s3_level or not trend_agree):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or trend disagreement
            if (close[i] > r3_level or not trend_agree):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals