#!/usr/bin/env python3
"""
12h Camarilla Pivot + Volume Spike + Chop Filter
Long when price touches L3 with volume spike in bullish chop regime
Short when price touches H3 with volume spike in bearish chop regime
Exit when price reaches opposite H3/L3 level or chop regime changes
Designed to work in ranging markets with clear institutional levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_chop_v1"
timeframe = "12h"
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
    
    # === 1d OHLC for Camarilla Pivot Calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    H3 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 6
    L3 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 6
    H4 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 2
    L4 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 2
    
    # Align to 12h timeframe
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # === Volume Spike Detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Volume at least 2x average
    
    # === Chop Filter (Choppiness Index) ===
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # First value
    
    # Sum of true ranges over 14 periods
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.where(
        (atr14 > 0) & (hh14 - ll14 > 0),
        100 * np.log10(atr14 / (hh14 - ll14)) / np.log10(14),
        50
    )
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(H3_12h[i]) or np.isnan(L3_12h[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches H3 or chop regime changes to trending
            if close[i] >= H3_12h[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 or chop regime changes to trending
            if close[i] <= L3_12h[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Chop regime: 38.2 <= chop <= 61.8 (ranging market)
            if 38.2 <= chop[i] <= 61.8:
                # Bullish chop: look for long at L3 with volume spike
                if close[i] <= L3_12h[i] * 1.001 and vol_spike[i]:
                    position = 1
                    signals[i] = 0.25
                # Bearish chop: look for short at H3 with volume spike
                elif close[i] >= H3_12h[i] * 0.999 and vol_spike[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals