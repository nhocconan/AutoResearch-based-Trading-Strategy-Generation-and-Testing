#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Pivot Point (Standard) breakout with volume confirmation and ATR stop
# - Calculate daily Pivot Point (PP) from prior day: PP = (H+L+C)/3
# - Calculate support/resistance: R1 = 2*PP - L, S1 = 2*PP - H, R2 = PP + (H-L), S2 = PP - (H-L)
# - Long when price closes above R1 with volume > 1.5x 20-period average
# - Short when price closes below S1 with volume > 1.5x 20-period average
# - Exit when price closes back through PP or ATR-based stop hit (2*ATR)
# - Uses 1d for pivot levels (stable) and 12h for execution
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d data for Pivot Point calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for stop loss (using 1d data)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Pivot Point and levels from prior day
    # PP = (H+L+C)/3
    # R1 = 2*PP - L, S1 = 2*PP - H
    # R2 = PP + (H-L), S2 = PP - (H-L)
    shift_high = np.roll(high_1d, 1)
    shift_low = np.roll(low_1d, 1)
    shift_close = np.roll(close_1d, 1)
    
    pp = (shift_high + shift_low + shift_close) / 3.0
    hl = shift_high - shift_low
    
    r1 = 2 * pp - shift_low
    s1 = 2 * pp - shift_high
    r2 = pp + hl
    s2 = pp - hl
    
    # Align pivot levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    pp_12h = align_htf_to_ltf(prices, df_1d, pp)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    
    # 12h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(pp_12h[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(atr_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price closes above R1 + volume surge
            if price > r1_12h[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price closes below S1 + volume surge
            elif price < s1_12h[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price closes below PP OR ATR stop hit (2*ATR)
            if price < pp_12h[i] or price < entry_price - 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above PP OR ATR stop hit (2*ATR)
            if price > pp_12h[i] or price > entry_price + 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_Point_R1S1_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0