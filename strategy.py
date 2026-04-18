#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA34_Volume_Filter
Hypothesis: Uses daily Camarilla pivot levels (R1/S1) for breakout entries, confirmed by 12h EMA34 trend and volume spikes. Designed to capture strong directional moves in both bull and bear markets by aligning with higher-timeframe trend and momentum. Tight entry conditions target 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and 12h data for EMA34
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_1d) < 2 or len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses same day's high
    prev_low[0] = low_1d[0]    # First day uses same day's low
    prev_close[0] = close_1d[0] # First day uses same day's close
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        k = 2 / (34 + 1)
        ema_34_12h[33] = np.mean(close_12h[:34])
        for i in range(34, len(close_12h)):
            ema_34_12h[i] = close_12h[i] * k + ema_34_12h[i-1] * (1 - k)
    
    # Align indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 60  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and above 12h EMA34
            if close[i] > r1_aligned[i] and vol_spike[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below S1 with volume spike and below 12h EMA34
            elif close[i] < s1_aligned[i] and vol_spike[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: minimum 2 bars hold, then exit on trend reversal or volatility drop
            if bars_since_entry >= 2:
                if close[i] < ema_34_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25  # Hold during minimum period
        
        elif position == -1:
            # Exit: minimum 2 bars hold, then exit on trend reversal or volatility drop
            if bars_since_entry >= 2:
                if close[i] > ema_34_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25  # Hold during minimum period
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA34_Volume_Filter"
timeframe = "4h"
leverage = 1.0