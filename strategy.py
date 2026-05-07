#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_S
Hypothesis: On 4h timeframe, use Camarilla pivot levels (R1/S1) from daily data for breakout entries, 
filtered by daily EMA trend and volume spikes to avoid false signals. 
Long when price breaks above R1 with volume spike and price above daily EMA. 
Short when price breaks below S1 with volume spike and price below daily EMA. 
Camarilla levels provide significant support/resistance, and combining with trend and volume 
filters reduces false breakouts. Works in both bull and bear markets by requiring alignment 
with daily trend.
"""
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_S"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R1, S1) from previous day's OHLC
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    # where C, H, L are from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Handle first day
    prev_close[0] = df_1d['close'].values[0]
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 2.0 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        if position == 0:
            # Minimum 8 bars between trades to reduce frequency (4h timeframe)
            if bars_since_entry < 8:
                continue
                
            # Long: price breaks above R1 + volume filter + price above EMA34
            if (close[i] > r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below S1 + volume filter + price below EMA34
            elif (close[i] < s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position != 0:
            # Exit: price returns to Camarilla midpoint or opposite breakout
            # Calculate Camarilla midpoint (P = close)
            prev_close_val = prev_close[i] if i < len(prev_close) else prev_close[-1]
            midpoint = prev_close_val
            
            if position == 1:
                if close[i] < midpoint:  # Price returns below midpoint
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > midpoint:  # Price returns above midpoint
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals