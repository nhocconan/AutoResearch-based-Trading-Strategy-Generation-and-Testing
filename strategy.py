#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_VolumeSpike_MultiTF_v1
12h timeframe, Camarilla Pivot levels from 1d, volume spike confirmation, and weekly trend filter.
Long when price breaks above R1 with volume spike and weekly trend up.
Short when price breaks below S1 with volume spike and weekly trend down.
Exit on opposite breakout or trend reversal.
Designed for low-frequency, high-conviction trades with minimal churn.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Camarilla Pivot Levels (from previous day) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    ph = df_1d['high'].values[:-1]  # previous day high
    pl = df_1d['low'].values[:-1]   # previous day low
    pc = df_1d['close'].values[:-1] # previous day close
    
    # Camarilla R1, S1 levels
    camarilla_r1 = pc + (ph - pl) * 1.1 / 12
    camarilla_s1 = pc - (ph - pl) * 1.1 / 12
    
    # Align to 12h timeframe (these levels are valid for the entire day)
    r1_12h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Volume Spike Detection (20-period average) ===
    vol_ma = np.zeros_like(volume)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count > 20:
            vol_sum -= volume[i - 20]
            vol_count -= 1
        if vol_count >= 20:
            vol_ma[i] = vol_sum / 20
        else:
            vol_ma[i] = np.nan
    
    volume_spike = volume > (vol_ma * 2.0)  # Volume at least 2x average
    
    # === Weekly Trend Filter (EMA34 on weekly close) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA34
    close_1w = df_1w['close'].values
    ema_1w = np.zeros_like(close_1w)
    alpha = 2 / (34 + 1)
    ema_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_1w[i] = ema_1w[i-1] + alpha * (close_1w[i] - ema_1w[i-1])
    
    # Align weekly EMA to 12h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Signal Generation ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have sufficient data for all indicators
    start_idx = max(20, len(df_1d) - 1)  # Need volume MA and at least 2 days for pivots
    
    for i in range(start_idx, n):
        # Skip if any data is unavailable
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: Price breaks above R1, volume spike, weekly uptrend
            if (close[i] > r1_12h[i] and 
                volume_spike[i] and 
                close[i] > ema_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below S1, volume spike, weekly downtrend
            elif (close[i] < s1_12h[i] and 
                  volume_spike[i] and 
                  close[i] < ema_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions
        elif position == 1:
            # Exit long: Price breaks below S1 OR weekly trend turns down
            if (close[i] < s1_12h[i] or 
                close[i] < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above R1 OR weekly trend turns up
            if (close[i] > r1_12h[i] or 
                close[i] > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_VolumeSpike_MultiTF_v1"
timeframe = "12h"
leverage = 1.0