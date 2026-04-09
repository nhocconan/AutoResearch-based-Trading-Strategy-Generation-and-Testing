#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v28"
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
    
    # Load daily data ONCE before loop for Camarilla levels
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    pp = np.full(len(df_d), np.nan)
    r4 = np.full(len(df_d), np.nan)
    s4 = np.full(len(df_d), np.nan)
    prev_high = np.full(len(df_d), np.nan)
    prev_low = np.full(len(df_d), np.nan)
    for i in range(1, len(df_d)):
        ph = df_d['high'].iloc[i-1]
        pl = df_d['low'].iloc[i-1]
        pc = df_d['close'].iloc[i-1]
        pp[i] = (ph + pl + pc) / 3.0
        r4[i] = pc + (ph - pl) * 1.1 / 2
        s4[i] = pc - (ph - pl) * 1.1 / 2
        prev_high[i] = ph
        prev_low[i] = pl
    
    # Align daily values to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_d, s4)
    prev_high_aligned = align_htf_to_ltf(prices, df_d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_d, prev_low)
    
    # Volume confirmation: 4-period average (4*4h = 16h ~ 2/3 day)
    vol_ma_4 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 4:
            vol_sum -= volume[i-4]
        if i >= 3:
            vol_ma_4[i] = vol_sum / 4
    
    # Calculate daily True Range for volatility filter
    tr = np.full(len(df_d), np.nan)
    atr = np.full(len(df_d), np.nan)
    for i in range(1, len(df_d)):
        high_low = df_d['high'].iloc[i] - df_d['low'].iloc[i]
        high_close = np.abs(df_d['high'].iloc[i] - df_d['close'].iloc[i-1])
        low_close = np.abs(df_d['low'].iloc[i] - df_d['close'].iloc[i-1])
        tr[i] = max(high_low, high_close, low_close)
        if i >= 14:
            atr[i] = np.mean(tr[i-13:i+1])
    
    atr_aligned = align_htf_to_ltf(prices, df_d, atr)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or 
            np.isnan(vol_ma_4[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back inside previous day's range
            if close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back inside previous day's range
            if close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above R4 with volume confirmation and volatility filter
            vol_ratio = volume[i] / vol_ma_4[i] if vol_ma_4[i] > 0 else 0
            if (close[i] > r4_aligned[i] and 
                vol_ratio > 1.5 and 
                atr_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below S4 with volume confirmation and volatility filter
            elif (close[i] < s4_aligned[i] and 
                  vol_ratio > 1.5 and 
                  atr_aligned[i] > 0):
                position = -1
                signals[i] = -0.25
    
    return signals