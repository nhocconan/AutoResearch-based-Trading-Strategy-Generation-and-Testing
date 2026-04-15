#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels (R1, S1) for breakout entries.
# In 1d uptrend (price > 1d EMA50), go long on break above R1 with volume confirmation.
# In 1d downtrend (price < 1d EMA50), go short on break below S1 with volume confirmation.
# Volume filter ensures breakout validity. Designed for low trade frequency (12-37/year) to minimize fee drag.
# Works in both bull and bear markets by following the 1d trend direction.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: EMA50 for trend ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.0 / 12)
    s1 = pivot - (range_1d * 1.0 / 12)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In 1d uptrend (price > 1d EMA50)
        # 2. Price breaks above 1d R1 level
        # 3. Volume confirmation
        if (close[i] > ema_50_1d_aligned[i]) and (close[i] > r1_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In 1d downtrend (price < 1d EMA50)
        # 2. Price breaks below 1d S1 level
        # 3. Volume confirmation
        elif (close[i] < ema_50_1d_aligned[i]) and (close[i] < s1_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_EMA50_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0