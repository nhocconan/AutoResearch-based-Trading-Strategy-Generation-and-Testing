#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h price reverses at 4h/1d pivot levels with volume confirmation.
# Uses 4h trend filter (EMA34) to avoid counter-trend trades. Session filter (08-20 UTC) reduces noise.
# Works in bull/bear by fading extremes at key levels with trend alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter and pivot calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h EMA34 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Daily pivot points from previous day
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Standard pivot calculation
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivots to 1h (use previous day's levels)
    pivot_1h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 35  # Need EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if outside session or missing data
        if not session_mask[i] or np.isnan(ema34_4h_aligned[i]) or np.isnan(pivot_1h[i]) or \
           np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or np.isnan(volume_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        if position == 0:
            # Long: price crosses above S1 with volume, above 4h EMA34 (uptrend bias)
            if close[i] > s1_1h[i] and volume_filter and close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price crosses below R1 with volume, below 4h EMA34 (downtrend bias)
            elif close[i] < r1_1h[i] and volume_filter and close[i] < ema34_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below pivot or reverses at R1
            if close[i] < pivot_1h[i] or close[i] > r1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above pivot or reverses at S1
            if close[i] > pivot_1h[i] or close[i] < s1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Pivot_Reversal_Volume_EMA34"
timeframe = "1h"
leverage = 1.0