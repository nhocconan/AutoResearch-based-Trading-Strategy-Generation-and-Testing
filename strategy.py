#!/usr/bin/env python3
"""
12h_1d_Camarilla_Reversal_v2
Hypothesis: On 12h timeframe, buy near L3 and sell near H3 of daily Camarilla pivot levels,
with volume confirmation and volatility regime filter. Designed for mean-reversion in ranging
markets and breakout strength in trending markets. Target: 15-30 trades/year (60-120 total).
Works in bull/bear via volatility regime and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Reversal_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's close for Camarilla calculation
    close_prev = np.concatenate([[close_1d[0]], close_1d[:-1]])
    range_1d = high_1d - low_1d
    
    h3 = close_prev + (range_1d * 1.1 / 4)
    l3 = close_prev - (range_1d * 1.1 / 4)
    h4 = close_prev + (range_1d * 1.1)
    l4 = close_prev - (range_1d * 1.1)
    
    # === WEEKLY VOLATILITY REGIME FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr_14 = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14:
            atr_14[i] = np.nan
        elif i == 14:
            atr_14[i] = np.nanmean(tr[1:15])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Volatility regime: low volatility = trending, high volatility = ranging
    vol_ma = np.full_like(atr_14, np.nan)
    for i in range(len(atr_14)):
        if i < 30:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.nanmean(atr_14[i-29:i+1])
    
    # High volatility regime (ranging) when current ATR > MA
    vol_regime = atr_14 > vol_ma
    
    # Align data to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1w, vol_regime.astype(float))
    
    # Volume average (20-period for 12h = ~10 days)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if indicators not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(vol_regime_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.3x average
        vol_confirm = volume[i] > 1.3 * vol_avg[i]
        
        # Only trade in high volatility (ranging) regime
        in_range_regime = vol_regime_aligned[i] > 0.5
        
        # Entry conditions: mean reversion at L3/H3
        long_setup = (close[i] <= l3_aligned[i]) and vol_confirm and in_range_regime
        short_setup = (close[i] >= h3_aligned[i]) and vol_confirm and in_range_regime
        
        # Exit conditions: mean reversion to opposite level or breakout
        exit_long = close[i] >= h3_aligned[i]
        exit_short = close[i] <= l3_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals