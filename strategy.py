#!/usr/bin/env python3
"""
1d_TRIX_ZeroCross_Volume_Trend
Hypothesis: Use daily TRIX (Triple Exponential Average) zero-line cross to capture momentum shifts, confirmed by volume spike (volume > 1.5x 20-day average) and filtered by 100-day EMA trend. Go long when TRIX crosses above zero with volume confirmation and price above EMA100, short when TRIX crosses below zero with volume confirmation and price below EMA100. TRIX is sensitive to trend changes and works in both bull (catching momentum) and bear (catching reversals) markets. Designed for 1d timeframe to limit trades (<20/year) and avoid fee drag.
"""

name = "1d_TRIX_ZeroCross_Volume_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate TRIX: triple EMA of log returns
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) then % change
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX as percentage change of triple EMA
    trix_raw = 100 * (pd.Series(ema3).pct_change().values)
    # Smooth TRIX with 12-period EMA (standard TRIX signal line smoothing)
    trix = pd.Series(trix_raw).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Align TRIX to daily timeframe (no extra delay needed)
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Get weekly EMA100 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_100_1w = pd.Series(df_1w['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    # Calculate volume average (20-day) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_aligned[i-1]) or 
            np.isnan(ema_100_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-day average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: TRIX crosses above zero (bullish momentum) + volume spike + price above weekly EMA100
            if trix_aligned[i-1] <= 0 and trix_aligned[i] > 0 and vol_spike and close[i] > ema_100_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero (bearish momentum) + volume spike + price below weekly EMA100
            elif trix_aligned[i-1] >= 0 and trix_aligned[i] < 0 and vol_spike and close[i] < ema_100_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or price breaks below EMA100
            if trix_aligned[i] < 0 or close[i] < ema_100_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or price breaks above EMA100
            if trix_aligned[i] > 0 or close[i] > ema_100_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals