#!/usr/bin/env python3
# 4h_1d_Pivot_R1_S1_Breakout_Volume_ATRFilter
# Hypothesis: Daily Camarilla R1/S1 breakouts on 4h timeframe with volume and ATR filter capture institutional moves while avoiding chop.
# Works in bull markets by catching breaks above R1; in bear markets by catching breaks below S1.
# Volume filter ensures institutional participation, ATR filter avoids low-volatility false breakouts.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_1d_Pivot_R1_S1_Breakout_Volume_ATRFilter"
timeframe = "4h"
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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = pivot + (high_1d - low_1d) * 1.1 / 12
    s1 = pivot - (high_1d - low_1d) * 1.1 / 12
    
    # Align daily Camarilla levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: volume > 1.8x 20-period EMA (more stringent to reduce trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (vol_ema20 * 1.8)
    
    # ATR filter: avoid low-volatility breakouts
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_filter = atr > (atr_ma * 0.8)  # Only trade when volatility is above 80% of its 50-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure ATR MA is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(atr_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + volume + volatility confirmation
            if close[i] > r1_aligned[i] and volume_filter[i] and atr_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume + volatility confirmation
            elif close[i] < s1_aligned[i] and volume_filter[i] and atr_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below pivot (mean reversion) or volatility drops
            if close[i] < pivot_aligned[i] or not atr_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above pivot (mean reversion) or volatility drops
            if close[i] > pivot_aligned[i] or not atr_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals