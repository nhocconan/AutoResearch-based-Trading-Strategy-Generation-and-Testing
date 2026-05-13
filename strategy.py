#!/usr/bin/env python3
# 4h_TRIX_Volume_Spike_1dTrend
# Hypothesis: TRIX (12) crosses zero with volume spike and 1d EMA50 trend filter.
# Long when TRIX crosses above zero in uptrend, short when crosses below zero in downtrend.
# Volume spike confirms institutional participation. Trend filter ensures alignment with higher timeframe momentum.
# Works in bull (TRIX crosses up in uptrend) and bear (TRIX crosses down in downtrend).
# Low frequency due to TRIX crossover requirement and volume confirmation.

name = "4h_TRIX_Volume_Spike_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for TRIX calculation and trend
    df_1d = get_htf_data(prices, '1d')
    
    # TRIX (12) on daily close: triple EMA of 1-period percent change
    close_1d = df_1d['close'].values
    # Calculate 1-period percent change
    pct_change = np.diff(close_1d) / close_1d[:-1]
    pct_change = np.insert(pct_change, 0, np.nan)  # align with original length
    # Triple EMA of percent change
    ema1 = pd.Series(pct_change).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.values * 100  # scale for readability
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 6-period average (1 day worth at 4h: 24h/4h=6)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > 2.0 * vol_ma_6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(trix_aligned[i-1]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + volume spike + daily uptrend
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + volume spike + daily downtrend
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and volume_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero OR trend reversal
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero OR trend reversal
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals