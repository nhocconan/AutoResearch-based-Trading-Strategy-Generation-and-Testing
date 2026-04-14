#!/usr/bin/env python3
"""
12h 1D Camarilla Pivot + Volume Spike + Choppiness Regime Filter
Long when price touches Camarilla L3 with volume spike in choppy market.
Short when price touches Camarilla H3 with volume spike in choppy market.
Exit on touch of opposite H3/L3 level or when choppiness drops below 38.2 (trending).
Designed for low turnover: ~15-30 trades/year per symbol.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: H4 = C + (H-L)*1.1/2, H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4, L4 = C - (H-L)*1.1/2
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use previous day's data
    rng = prev_high - prev_low
    H3 = prev_close + rng * 1.1 / 4
    L3 = prev_close - rng * 1.1 / 4
    
    # Align to 12h timeframe
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    
    # Choppiness index on daily (need OHLC)
    # Chop = 100 * log10(sum(ATR(14)) / (n * (highest-high - lowest-low))) / log10(n)
    # Simplified: use rolling std dev of returns as proxy for chop
    returns = np.log(df_1d['close'] / df_1d['close'].shift(1))
    chop_raw = returns.rolling(window=14, min_periods=14).std().values
    # Normalize to 0-100 scale (approximation)
    chop_raw = np.nan_to_num(chop_raw, nan=0.0)
    chop_max = np.maximum.accumulate(chop_raw)
    chop = 100 * chop_raw / (chop_max + 1e-10)
    chop = np.nan_to_num(chop, nan=50.0)
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if chop data not ready
        if np.isnan(chop_12h[i]):
            continue
            
        # Only trade in choppy market (Chop > 61.8 = ranging)
        if chop_12h[i] <= 61.8:
            # If trending, exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price at L3 with volume spike
            if low[i] <= L3_12h[i] and vol_spike[i]:
                position = 1
                signals[i] = position_size
            # Short: price at H3 with volume spike
            elif high[i] >= H3_12h[i] and vol_spike[i]:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit long: price touches H3 or chop drops
            if high[i] >= H3_12h[i] or chop_12h[i] < 61.8:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit short: price touches L3 or chop drops
            if low[i] <= L3_12h[i] or chop_12h[i] < 61.8:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1D_Camarilla_Pivot_Volume_Chop"
timeframe = "12h"
leverage = 1.0