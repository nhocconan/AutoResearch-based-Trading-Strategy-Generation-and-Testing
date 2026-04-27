#!/usr/bin/env python3
"""
12h CAMARILLA PIVOT REVERSION WITH VOLUME FILTER.
Long when price touches S1 (support) and reverses up with volume confirmation.
Short when price touches R1 (resistance) and reverses down with volume confirmation.
Exit when price crosses central pivot (PP) or after 6 bars to prevent overtrading.
Uses 1d CAMARILLA levels (PP, S1, R1) calculated from prior day's H-L-C.
Designed to generate 12-37 trades/year per symbol with mean-reversion edge in ranging markets.
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
    
    # Get 1d data for CAMARILLA pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate CAMARILLA levels from prior day's OHLC
    # PP = (H + L + C) / 3
    # S1 = C - (H - L) * 1.1 / 12
    # R1 = C + (H - L) * 1.1 / 12
    # Using prior day's values (shifted by 1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate levels for each day
    pp = (high_1d + low_1d + close_1d) / 3.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    
    # Align to 12h timeframe (use prior day's levels for current day)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Volume filter: volume > 1.3x average (to confirm reversal)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Maximum hold period to prevent overtrading (6 bars = 3 days)
    max_hold_bars = 6
    
    # Warmup: need 20-period volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        pp_val = pp_aligned[i]
        s1_val = s1_aligned[i]
        r1_val = r1_aligned[i]
        
        # Volume filter
        vol_filter = vol_now > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: price touches S1 (<= S1 * 1.005) and reverses up + volume filter
            if price_now <= s1_val * 1.005 and price_now > s1_val * 0.995 and vol_filter:
                # Additional confirmation: price above open (bullish reversal)
                if price_now > prices['open'].iloc[i]:
                    signals[i] = size
                    position = 1
                    bars_held = 0
            # Short: price touches R1 (>= R1 * 0.995) and reverses down + volume filter
            elif price_now >= r1_val * 0.995 and price_now < r1_val * 1.005 and vol_filter:
                # Additional confirmation: price below open (bearish reversal)
                if price_now < prices['open'].iloc[i]:
                    signals[i] = -size
                    position = -1
                    bars_held = 0
            else:
                signals[i] = 0.0
                bars_held = 0
        elif position == 1:
            bars_held += 1
            # Exit conditions:
            # 1. Price crosses above PP (mean reversion complete)
            # 2. Maximum hold period reached
            if price_now > pp_val or bars_held >= max_hold_bars:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            bars_held += 1
            # Exit conditions:
            # 1. Price crosses below PP (mean reversion complete)
            # 2. Maximum hold period reached
            if price_now < pp_val or bars_held >= max_hold_bars:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_Pivot_Reversion_Volume_Filter"
timeframe = "12h"
leverage = 1.0