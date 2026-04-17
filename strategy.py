#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_Breakout_Volume_Regime_v1
Breakout above Camarilla R1 or below S1 with volume spike and Choppiness regime filter.
Exit when price returns to pivot (close). Uses 1d trend filter.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # === Pivot Point and Support/Resistance Levels ===
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    
    # === Volume Spike: volume > 1.5 * 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    # === Choppiness Index: CHOP > 61.8 = ranging (favor mean reversion) ===
    def true_range(h, l, c):
        tr1 = h - l
        tr2 = np.abs(h - np.roll(c, 1))
        tr3 = np.abs(l - np.roll(c, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        return tr
    
    tr = true_range(high, low, close)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(tr, axis=0) / np.log10(14)) / np.log10((hh14 - ll14) / atr14) if False else \
           100 * np.log10(pd.Series(tr).rolling(window=14, min_periods=14).sum().values / 
                          (pd.Series(hh14 - ll14).rolling(window=14, min_periods=14).mean().values / atr14 + 1e-10))
    # Simplified Choppiness: 100 * log14(sum(TR14) / (14 * ATR14))
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(tr_sum / (14 * atr14 + 1e-10)) / np.log10(14)
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot[i]) or 
            np.isnan(r1[i]) or 
            np.isnan(s1[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1, volume spike, chop < 61.8 (trending), price above 1d EMA50
            if (close[i] > r1[i] and 
                vol_spike[i] and 
                chop[i] < 61.8 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1, volume spike, chop < 61.8 (trending), price below 1d EMA50
            elif (close[i] < s1[i] and 
                  vol_spike[i] and 
                  chop[i] < 61.8 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to pivot (close)
        elif position == 1:
            # Exit long: price crosses below pivot
            if close[i] < pivot[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above pivot
            if close[i] > pivot[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0