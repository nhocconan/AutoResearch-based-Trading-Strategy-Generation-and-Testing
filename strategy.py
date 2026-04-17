#!/usr/bin/env python3
"""
6h_MultiTF_Pivot_Reversal_v1
Daily Pivot Point reversal with weekly trend filter and volume confirmation.
- Daily Pivot (PP), Support 1 (S1), Resistance 1 (R1) calculated from prior day
- Weekly trend: price above/below weekly EMA20 determines long/short bias
- Entry: price rejects S1/R1 (touch + close back inside) in direction of weekly trend
- Volume confirmation: volume > 1.5x 20-period average
- Target: 50-150 total trades over 4 years (12-37/year)
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
    
    # === Daily Pivot Points (from prior day) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points for each day
    pp = np.zeros_like(close_1d)
    s1 = np.zeros_like(close_1d)
    r1 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Use prior day's OHLC
        pp[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        s1[i] = 2.0 * pp[i] - high_1d[i-1]
        r1[i] = 2.0 * pp[i] - low_1d[i-1]
    
    # === Weekly EMA20 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 on weekly
    ema_20 = np.zeros_like(close_1w)
    if len(close_1w) >= 20:
        ema_20[19] = np.mean(close_1w[:20])
        multiplier = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema_20[i] = (close_1w[i] - ema_20[i-1]) * multiplier + ema_20[i-1]
    else:
        ema_20[:] = close_1w[0] if len(close_1w) > 0 else 0
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5
    
    # === Align to 6h timeframe ===
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long bias: price above weekly EMA20
            if close[i] > ema_20_aligned[i]:
                # Look for rejection of S1: low touches S1 and close returns above it
                if (low[i] <= s1_aligned[i] and close[i] > s1_aligned[i] and vol_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                    continue
            # Short bias: price below weekly EMA20
            elif close[i] < ema_20_aligned[i]:
                # Look for rejection of R1: high touches R1 and close returns below it
                if (high[i] >= r1_aligned[i] and close[i] < r1_aligned[i] and vol_confirm[i]):
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below S1 or weekly trend changes
            if close[i] < s1_aligned[i] or close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above R1 or weekly trend changes
            if close[i] > r1_aligned[i] or close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_MultiTF_Pivot_Reversal_v1"
timeframe = "6h"
leverage = 1.0