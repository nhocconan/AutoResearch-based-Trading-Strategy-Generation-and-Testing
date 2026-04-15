#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot R1/S1 breakout with volume confirmation and choppiness regime filter.
# In ranging markets (chop > 61.8), buy at S1 breakout with volume, sell at R1 breakdown with volume.
# Uses discrete position sizing (0.25) to minimize fee drift. Designed for low trade frequency (12-25/year)
# to avoid fee drag while capturing mean reversion in choppy regimes and avoiding false breakouts in trends.

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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1d Indicators: Choppiness Index (14) ===
    atr_1d = []
    for i in range(len(df_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        atr_1d.append(tr)
    atr_1d = np.array(atr_1d)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / np.log(14) / (max_hh - min_ll))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Choppy market (chop > 61.8 = ranging)
        # 2. Price breaks above S1 with volume
        # 3. Close above S1 (confirm breakout)
        if (chop_aligned[i] > 61.8) and (close[i] > s1_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Choppy market (chop > 61.8 = ranging)
        # 2. Price breaks below R1 with volume
        # 3. Close below R1 (confirm breakdown)
        elif (chop_aligned[i] > 61.8) and (close[i] < r1_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_Volume_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0