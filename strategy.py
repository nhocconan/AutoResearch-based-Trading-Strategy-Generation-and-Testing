# 6h_Wyckoff_Spring_UpThrust_Volume_Confirmation
# Hypothesis: Wyckoff method identifies accumulation (spring) and distribution (upthrust) patterns.
# A spring occurs when price tests below a recent low but closes back above with strong volume,
# indicating accumulation and potential trend reversal up. An upthrust is the opposite - price
# tests above a recent high but closes back below with strong volume, indicating distribution
# and potential trend reversal down. This strategy identifies these patterns on 6h timeframe
# with volume confirmation and uses 12h trend filter to align with higher timeframe momentum.
# Works in both bull and bear markets by capturing reversal points at key supply/demand zones.
# Targets 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.25).

name = "6h_Wyckoff_Spring_UpThrust_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    close_12h = df_12h['close'].values

    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Calculate 20-period rolling high and low for Wyckoff patterns
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Spring pattern - price tests below recent low but closes back above with volume
            # Spring: low penetrates below 20-period low but close > 20-period low
            spring = low[i] < low_min_20[i] and close[i] > low_min_20[i]
            # Volume confirmation: volume > 1.5x average
            volume_confirm = volume[i] > vol_avg_20[i] * 1.5
            # Trend filter: price above 12h EMA50 for long bias
            trend_filter = close[i] > ema50_12h_aligned[i]
            
            if spring and volume_confirm and trend_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Upthrust pattern - price tests above recent high but closes back below with volume
            # Upthrust: high penetrates above 20-period high but close < 20-period high
            upthrust = high[i] > high_max_20[i] and close[i] < high_max_20[i]
            # Volume confirmation: volume > 1.5x average
            volume_confirm = volume[i] > vol_avg_20[i] * 1.5
            # Trend filter: price below 12h EMA50 for short bias
            trend_filter = close[i] < ema50_12h_aligned[i]
            
            if upthrust and volume_confirm and trend_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                
        elif position == 1:
            # EXIT LONG: Price breaks below 20-period low OR trend turns bearish
            if close[i] < low_min_20[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-period high OR trend turns bullish
            if close[i] > high_max_20[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals