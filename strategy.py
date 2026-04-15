#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d volume spike and 1w EMA trend filter.
# Uses 1d Camarilla levels (R1, S1) for institutional reference points, 
# 1d volume > 2.0x 20-bar SMA for institutional participation confirmation,
# and 1w EMA(50) for long-term trend alignment to avoid counter-trend trades.
# Designed for very low trade frequency (12-37/year) with discrete position sizing (0.25).
# Works in bull/bear: 1w EMA prevents fighting the trend, volume spike ensures momentum,
# Camarilla levels provide high-probability reversal/breakout zones.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1d Indicators: Volume Spike Filter ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_sma_20 * 2.0)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(50) for long-term trend bias
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1 with volume spike
        # 2. 1w price above EMA50 (bullish long-term trend bias)
        if (close[i] > r1_aligned[i] and
            vol_spike[i] and
            close[i] > ema_50_1w_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1 with volume spike
        # 2. 1w price below EMA50 (bearish long-term trend bias)
        elif (close[i] < s1_aligned[i] and
              vol_spike[i] and
              close[i] < ema_50_1w_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_VolumeSpike_EMA50_v1"
timeframe = "12h"
leverage = 1.0