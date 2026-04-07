#!/usr/bin/env python3
"""
6H Weekly Pivot Reversion with Volume Confirmation and 1D Trend Filter
Mean reversion at weekly support/resistance levels with volume exhaustion and 1D trend alignment.
Long: price near weekly S1/S2 with bullish 1D EMA and volume drying up
Short: price near weekly R1/R2 with bearish 1D EMA and volume drying up
Exit: price returns to weekly pivot or opposite extreme
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_reversion_volume_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly pivot points (using prior week) ===
    df_1w = get_htf_data(prices, '1w')
    # Calculate pivot from previous week's OHLC (already shifted in align_htf_to_ltf)
    pivot = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3
    width = df_1w['high'].values - df_1w['low'].values
    r1 = 2 * pivot - df_1w['low'].values
    s1 = 2 * pivot - df_1w['high'].values
    r2 = pivot + width
    s2 = pivot - width
    
    # Align to 6h timeframe (already shifted by 1 week in align_htf_to_ltf)
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    
    # === Volume exhaustion signal ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 4-day average
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 1D trend filter (EMA 50) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to pivot or reaches S2
            if close[i] >= pivot_6h[i] or close[i] <= s2_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to pivot or reaches R2
            if close[i] <= pivot_6h[i] or close[i] >= r2_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume exhaustion: below average volume
            if vol_ratio[i] > 0.8:
                signals[i] = 0.0
                continue
            
            # Entry conditions with 1D trend filter
            # Long: price near support in uptrend
            if (close[i] <= s1_6h[i] * 1.02 and close[i] >= s2_6h[i] * 0.98 and 
                ema_1d_aligned[i] > ema_1d_aligned[i-1]):
                position = 1
                signals[i] = 0.25
            # Short: price near resistance in downtrend
            elif (close[i] >= r1_6h[i] * 0.98 and close[i] <= r2_6h[i] * 1.02 and 
                  ema_1d_aligned[i] < ema_1d_aligned[i-1]):
                position = -1
                signals[i] = -0.25
    
    return signals