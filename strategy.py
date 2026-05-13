#!/usr/bin/env python3
# 1D_Weekly_Pivot_Breakout_1wTrend_Volume
# Hypothesis: Use weekly pivot point resistance/support for breakout entries with weekly trend filter and volume confirmation.
# Long when price breaks above weekly R1 in uptrend with volume spike, short when price breaks below weekly S1 in downtrend with volume spike.
# Weekly pivot points provide strong support/resistance; weekly trend filter avoids counter-trend trades; volume confirms breakout strength.
# Designed for low trade frequency (30-100 total over 4 years) with clear entry/exit rules to avoid overtrading.

name = "1D_Weekly_Pivot_Breakout_1wTrend_Volume"
timeframe = "1d"
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

    # Get weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points based on previous week's OHLC
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    close_1w = df_1w['close']
    
    # Weekly pivot point and support/resistance levels
    pp_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pp_1w - low_1w  # Resistance 1
    s1_1w = 2 * pp_1w - high_1w  # Support 1
    
    # Weekly EMA for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly data to daily timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w.values)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w.values)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w.values)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Volume filter: >1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + price above weekly EMA50 (uptrend) + volume spike
            if (close[i] > r1_1w_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + price below weekly EMA50 (downtrend) + volume spike
            elif (close[i] < s1_1w_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point or trend changes (price below weekly EMA50)
            if (close[i] <= pp_1w_aligned[i] or close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point or trend changes (price above weekly EMA50)
            if (close[i] >= pp_1w_aligned[i] or close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals