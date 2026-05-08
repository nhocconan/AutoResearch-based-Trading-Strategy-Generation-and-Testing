#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R4 (resistance 4) AND 1d EMA34 rising AND volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S4 (support 4) AND 1d EMA34 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Camarilla H-L range (between H4 and L4).
# Camarilla levels provide high-probability reversal/breakout levels based on prior day's range.
# The 1d EMA34 filter ensures we trade with the higher timeframe trend.
# Volume spike confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Camarilla_R4_S4_Breakout_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 1d data (based on previous day's range)
    # Using previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    # Resistance levels
    r1 = pivot + (range_hl * 1.0 / 12)
    r2 = pivot + (range_hl * 2.0 / 12)
    r3 = pivot + (range_hl * 3.0 / 12)
    r4 = pivot + (range_hl * 4.0 / 12)
    # Support levels
    s1 = pivot - (range_hl * 1.0 / 12)
    s2 = pivot - (range_hl * 2.0 / 12)
    s3 = pivot - (range_hl * 3.0 / 12)
    s4 = pivot - (range_hl * 4.0 / 12)
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    h4_aligned = align_htf_to_ltf(prices, df_1d, r4)  # H4 is same as R4 for exit
    l4_aligned = align_htf_to_ltf(prices, df_1d, s4)  # L4 is same as S4 for exit
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 35)  # Sufficient warmup for EMA34 and Camarilla
    
    for i in range(start_idx, n):
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R4, 1d EMA34 rising, volume filter
            long_cond = (close[i] > r4_aligned[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price breaks below Camarilla S4, 1d EMA34 falling, volume filter
            short_cond = (close[i] < s4_aligned[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Camarilla L4 (support 4)
            if close[i] < l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Camarilla H4 (resistance 4)
            if close[i] > h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals