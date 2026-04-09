#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    r4 = np.full(len(df_1d), np.nan)
    s4 = np.full(len(df_1d), np.nan)
    prev_high = np.full(len(df_1d), np.nan)
    prev_low = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        ph = float(df_1d['high'].iloc[i-1])
        pl = float(df_1d['low'].iloc[i-1])
        pc = float(df_1d['close'].iloc[i-1])
        r4[i] = pc + (ph - pl) * 1.1 / 2
        s4[i] = pc - (ph - pl) * 1.1 / 2
        prev_high[i] = ph
        prev_low[i] = pl
    
    # Align 1d values to 12h timeframe
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    prev_high_12h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_12h = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Volume confirmation: 4-period average (48h)
    vol_ma_4 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 4:
            vol_sum -= volume[i-4]
        if i >= 3:
            vol_ma_4[i] = vol_sum / 4
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_12h[i]) or 
            np.isnan(s4_12h[i]) or 
            np.isnan(prev_high_12h[i]) or 
            np.isnan(prev_low_12h[i]) or 
            np.isnan(vol_ma_4[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back inside previous day's range
            if close[i] <= prev_high_12h[i] and close[i] >= prev_low_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back inside previous day's range
            if close[i] <= prev_high_12h[i] and close[i] >= prev_low_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above R4 with volume confirmation
            vol_ratio = volume[i] / vol_ma_4[i] if vol_ma_4[i] > 0 else 0
            if close[i] > r4_12h[i] and vol_ratio > 2.0:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below S4 with volume confirmation
            elif close[i] < s4_12h[i] and vol_ratio > 2.0:
                position = -1
                signals[i] = -0.25
    
    return signals