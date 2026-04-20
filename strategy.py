#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Regime_v1
Concept: 12h Camarilla pivot breakout with daily volume confirmation and chop regime filter.
- Long: Close > R1 and volume > 1.5x 20-period average and CHOP(14) < 38.2 (trending)
- Short: Close < S1 and volume > 1.5x 20-period average and CHOP(14) < 38.2 (trending)
- Exit: Close crosses back below R1 (long) or above S1 (short)
- Position sizing: 0.25
- Target: 12-37 trades/year (50-150 total over 4 years)
- Works in bull/bear: Pivot levels define support/resistance, volume confirms breakout, chop filter avoids whipsaws in ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Pivot_R1S1_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivots and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily: Pivot points (using previous day's OHLC) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align to 12h timeframe (use previous day's levels for today's trading)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Daily: Choppiness Index (CHOP) for regime filter ===
    # CHOP = 100 * log10(sum(ATR over n) / (n * (highest high - lowest low over n)))
    # Simplified: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
    high_low = np.maximum(high_1d, np.roll(high_1d, 1))
    high_low[0] = high_1d[0]  # first value
    true_range = np.maximum(
        high_low - np.roll(low_1d, 1),
        np.abs(close_1d - np.roll(close_1d, 1))
    )
    true_range[0] = high_1d[0] - low_1d[0]  # first TR
    
    atr14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    highest_high14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range14 = highest_high14 - lowest_low14
    range14[range14 == 0] = 1e-10
    
    chop = 100 * np.log10(pd.Series(atr14).rolling(window=14, min_periods=14).sum().values / (14 * range14))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === Volume confirmation (20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_val = volume[i]
        vol_ma20_val = vol_ma20[i]
        chop_val = chop_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(vol_val) or np.isnan(vol_ma20_val) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume confirmation and trending market
            if close_val > r1_val and vol_val > 1.5 * vol_ma20_val and chop_val < 38.2:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume confirmation and trending market
            elif close_val < s1_val and vol_val > 1.5 * vol_ma20_val and chop_val < 38.2:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close back below R1
            if close_val < r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close back above S1
            if close_val > s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals