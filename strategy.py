#/usr/bin/env python3
# 1d_PivotBounce_TrendFilter
# Hypothesis: Price reversals at weekly pivot points (R1/S1) with trend alignment (weekly EMA20) provide high-probability entries.
# Uses mean reversion in ranging markets and trend continuation in trending markets. Designed for 1d timeframe to target 15-25 trades per year.
# Works in bull/bear via trend filter - only takes long in uptrend, short in downtrend.

name = "1d_PivotBounce_TrendFilter"
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

    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # Calculate weekly pivot points (using prior week's OHLC)
    # Standard formula: P = (H + L + C)/3, R1 = 2P - L, S1 = 2P - H
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    # Pivot points based on previous week
    pivot = (np.roll(wk_high, 1) + np.roll(wk_low, 1) + np.roll(wk_close, 1)) / 3.0
    r1 = 2 * pivot - np.roll(wk_low, 1)
    s1 = 2 * pivot - np.roll(wk_high, 1)
    
    # Set first week values to avoid roll issues
    pivot[0] = (wk_high[0] + wk_low[0] + wk_close[0]) / 3.0
    r1[0] = 2 * pivot[0] - wk_low[0]
    s1[0] = 2 * pivot[0] - wk_high[0]
    
    # Weekly EMA20 for trend filter
    wk_ema20 = pd.Series(wk_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly data to daily
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    wk_ema20_aligned = align_htf_to_ltf(prices, df_1w, wk_ema20)

    # Volume confirmation: 1.5x 20-day SMA (moderate threshold)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after EMA needs 20 bars
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(wk_ema20_aligned[i]) or 
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at S1 support with weekly uptrend and volume
            if (low[i] <= s1_aligned[i] * 1.002 and  # Allow 0.2% slippage
                close[i] > wk_ema20_aligned[i] and
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R1 resistance with weekly downtrend and volume
            elif (high[i] >= r1_aligned[i] * 0.998 and  # Allow 0.2% slippage
                  close[i] < wk_ema20_aligned[i] and
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot or trend breaks
            if high[i] >= pivot_aligned[i] * 0.998 or close[i] < wk_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot or trend breaks
            if low[i] <= pivot_aligned[i] * 1.002 or close[i] > wk_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals