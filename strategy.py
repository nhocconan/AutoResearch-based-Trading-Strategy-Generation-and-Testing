#!/usr/bin/env python3
name = "1h_4h1d_Camarilla_Pivot_Breakout"
timeframe = "1h"
leverage = 1.0

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
    
    # Load 4h data ONCE before loop for Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot points (using previous 4h bar)
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels (using standard formula)
    h5 = pivot + (range_hl * 1.1 / 2)  # Resistance 5
    h4 = pivot + (range_hl * 1.1 / 4)  # Resistance 4
    h3 = pivot + (range_hl * 1.1 / 6)  # Resistance 3
    l3 = pivot - (range_hl * 1.1 / 6)  # Support 3
    l4 = pivot - (range_hl * 1.1 / 4)  # Support 4
    l5 = pivot - (range_hl * 1.1 / 2)  # Support 5
    
    # Align 4h Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 24-period average (1 day of 1h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above L3 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > l3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: price below H3 with volume and daily downtrend
            elif close[i] < h3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price back below L3 or volume drops
            if close[i] < l3_aligned[i] or volume[i] < vol_ma_24[i] * 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above H3 or volume drops
            if close[i] > h3_aligned[i] or volume[i] < vol_ma_24[i] * 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla L3/H3 breakout with 1d trend and volume confirmation
# - 4h Camarilla L3/H3 act as key support/resistance levels from prior 4h bar
# - Breakout above L3 with volume in daily uptrend = long opportunity
# - Breakdown below H3 with volume in daily downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Works in both bull (buy L3 breaks in uptrend) and bear (sell H3 breaks in downtrend)
# - Exit when price returns to L3/H3 or volume weakens
# - Position size 0.20 targets ~20-50 trades/year, avoiding fee drag
# - Uses 4h Camarilla levels (not daily) for better stability on 1h timeframe
# - Daily trend filter reduces whipsaws vs using same timeframe
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Novel combination: 4h Camarilla (4h) + trend (1d) + volume (1h) not recently tried
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits