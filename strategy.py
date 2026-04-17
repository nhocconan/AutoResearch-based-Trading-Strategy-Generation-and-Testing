#!/usr/bin/env python3
"""
4h_12h_Camarilla_R1_S1_Breakout_Volume_Trend_v1
4-hour strategy using 12-hour Camarilla pivot levels (R1/S1) with volume confirmation and trend filter.
Enters long when price breaks above R1 with volume above average and price above 12h EMA34.
Enters short when price breaks below S1 with volume above average and price below 12h EMA34.
Uses tight entry conditions to limit trades and avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 12-hour Camarilla Pivot Levels (R1, S1) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point and Camarilla levels
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    r1_12h = close_12h + (high_12h - low_12h) * 1.1 / 12
    s1_12h = close_12h - (high_12h - low_12h) * 1.1 / 12
    
    # Align to 4h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # === 12-hour EMA34 for Trend Filter ===
    ema34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # === 12-hour Volume for Confirmation ===
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 80
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_12h_aligned[i]) or 
            np.isnan(s1_12h_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 12h bar's volume for confirmation
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        vol_confirmed = vol_12h_current > 1.5 * vol_ma_12h_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > r1_12h_aligned[i]
        breakout_short = close[i] < s1_12h_aligned[i]
        
        # Exit conditions: return to opposite pivot level
        exit_long = close[i] < s1_12h_aligned[i]
        exit_short = close[i] > r1_12h_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above R1 with volume confirmation and trend filter
            if breakout_long and vol_confirmed and close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below S1 with volume confirmation and trend filter
            elif breakout_short and vol_confirmed and close[i] < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below S1
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Camarilla_R1_S1_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0