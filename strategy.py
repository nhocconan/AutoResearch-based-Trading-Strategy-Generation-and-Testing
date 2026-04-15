#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with weekly trend filter and volume confirmation.
# Uses 1w EMA(50) for long-term trend bias and 1d Camarilla pivot levels (R1/S1) for entry timing.
# Includes volume filter (current volume > 1.3x 20-bar SMA) to avoid low-momentum breakouts.
# Designed for low trade frequency (12-37/year) to minimize fee drag in choppy markets.
# Works in bull/bear: 1w EMA avoids counter-trend trades, Camarilla breakout captures momentum with structure.

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
    
    # === 1d Indicators: Camarilla Pivot Points (R1, S1) ===
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    pivot = (high_1d + low_1d + close_1d) / 3
    rng = high_1d - low_1d
    r1 = pivot + (rng * 1.1 / 12)
    s1 = pivot - (rng * 1.1 / 12)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(50) for long-term trend bias
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1
        # 2. 1w price above EMA50 (bullish long-term trend bias)
        # 3. Volume confirmation
        if (close[i] > r1_aligned[i] and
            close[i] > ema_50_1w_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1
        # 2. 1w price below EMA50 (bearish long-term trend bias)
        # 3. Volume confirmation
        elif (close[i] < s1_aligned[i] and
              close[i] < ema_50_1w_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R1S1_1wEMA50_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0