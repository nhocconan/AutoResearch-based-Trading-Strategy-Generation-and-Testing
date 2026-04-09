#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for 12h ATR and 12h close
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h ATR(14) for volatility filter
    tr_12h = np.full(len(df_12h), np.nan)
    atr_12h = np.full(len(df_12h), np.nan)
    for i in range(1, len(df_12h)):
        high_low = df_12h['high'].iloc[i] - df_12h['low'].iloc[i]
        high_close = np.abs(df_12h['high'].iloc[i] - df_12h['close'].iloc[i-1])
        low_close = np.abs(df_12h['low'].iloc[i] - df_12h['close'].iloc[i-1])
        tr_12h[i] = max(high_low, high_close, low_close)
        if i >= 13:
            atr_12h[i] = np.mean(tr_12h[i-13:i+1])
    
    # Align 12h ATR to 4h timeframe
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Load daily data ONCE before loop for Camarilla levels (using prior day's OHLC)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    r4 = np.full(len(df_d), np.nan)
    s4 = np.full(len(df_d), np.nan)
    prev_high = np.full(len(df_d), np.nan)
    prev_low = np.full(len(df_d), np.nan)
    for i in range(1, len(df_d)):
        ph = df_d['high'].iloc[i-1]
        pl = df_d['low'].iloc[i-1]
        pc = df_d['close'].iloc[i-1]
        r4[i] = pc + (ph - pl) * 1.1 / 2
        s4[i] = pc - (ph - pl) * 1.1 / 2
        prev_high[i] = ph
        prev_low[i] = pl
    
    # Align daily values to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_d, s4)
    prev_high_aligned = align_htf_to_ltf(prices, df_d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_d, prev_low)
    
    # Volume confirmation: 3-period average (3*4h = 12h) - matches 12h ATR
    vol_ma_3 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 3:
            vol_sum -= volume[i-3]
        if i >= 2:
            vol_ma_3[i] = vol_sum / 3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or 
            np.isnan(vol_ma_3[i]) or 
            np.isnan(atr_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back inside previous day's range
            if close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes back inside previous day's range
            if close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Enter long: price closes above R4 with volume confirmation and volatility filter
            vol_ratio = volume[i] / vol_ma_3[i] if vol_ma_3[i] > 0 else 0
            if (close[i] > r4_aligned[i] and 
                vol_ratio > 1.5 and 
                atr_12h_aligned[i] > 0):
                position = 1
                signals[i] = 0.30
            # Enter short: price closes below S4 with volume confirmation and volatility filter
            elif (close[i] < s4_aligned[i] and 
                  vol_ratio > 1.5 and 
                  atr_12h_aligned[i] > 0):
                position = -1
                signals[i] = -0.30
    
    return signals