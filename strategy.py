#!/usr/bin/env python3
# 4h_TRIX_Volume_Spike_Trend_Filter
# Hypothesis: Use TRIX (triple-smoothed EMA) on 4h to detect momentum shifts, confirmed by volume spike and 1d EMA trend. TRIX reduces noise and is effective in both bull and bear markets by filtering out minor fluctuations. Volume spike confirms institutional participation. Trend filter ensures alignment with higher timeframe momentum, reducing false signals in choppy markets. Designed for low frequency to avoid fee drag.

name = "4h_TRIX_Volume_Spike_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for TRIX calculation
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # TRIX calculation: triple EMA of percentage change
    # TRIX = EMA(EMA(EMA(close, period), period), period) 
    # We use period=12 as standard
    period = 12
    ema1 = pd.Series(close_4h).ewm(span=period, adjust=False, min_periods=period).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False, min_periods=period).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, adjust=False, min_periods=period).mean().values
    # Calculate percentage change of the triple EMA
    pct_change = np.zeros_like(ema3)
    pct_change[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # Final TRIX: smooth the percentage change
    trix = pd.Series(pct_change).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_4h, trix)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 6-period average (1.5 days worth at 4h)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > 2.0 * vol_ma_6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + volume spike + 1d uptrend
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + volume spike + 1d downtrend
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