#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Camarilla levels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (using prior week's OHLC)
    pp = np.full(len(df_w), np.nan)
    r4 = np.full(len(df_w), np.nan)
    s4 = np.full(len(df_w), np.nan)
    prev_high = np.full(len(df_w), np.nan)
    prev_low = np.full(len(df_w), np.nan)
    for i in range(1, len(df_w)):
        ph = df_w['high'].iloc[i-1]
        pl = df_w['low'].iloc[i-1]
        pc = df_w['close'].iloc[i-1]
        pp[i] = (ph + pl + pc) / 3.0
        r4[i] = pc + (ph - pl) * 1.1 / 2
        s4[i] = pc - (ph - pl) * 1.1 / 2
        prev_high[i] = ph
        prev_low[i] = pl
    
    # Align weekly values to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_w, pp)
    r4_aligned = align_htf_to_ltf(prices, df_w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_w, s4)
    prev_high_aligned = align_htf_to_ltf(prices, df_w, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_w, prev_low)
    
    # Volume confirmation: 3-day average
    vol_ma_3 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 3:
            vol_sum -= volume[i-3]
        if i >= 2:
            vol_ma_3[i] = vol_sum / 3
    
    # Calculate weekly ATR for volatility filter
    tr = np.full(len(df_w), np.nan)
    atr = np.full(len(df_w), np.nan)
    for i in range(1, len(df_w)):
        high_low = df_w['high'].iloc[i] - df_w['low'].iloc[i]
        high_close = np.abs(df_w['high'].iloc[i] - df_w['close'].iloc[i-1])
        low_close = np.abs(df_w['low'].iloc[i] - df_w['close'].iloc[i-1])
        tr[i] = max(high_low, high_close, low_close)
        if i >= 10:
            atr[i] = np.mean(tr[i-9:i+1])
    
    atr_aligned = align_htf_to_ltf(prices, df_w, atr)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or 
            np.isnan(vol_ma_3[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back inside previous week's range
            if close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back inside previous week's range
            if close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above R4 with volume confirmation and volatility filter
            vol_ratio = volume[i] / vol_ma_3[i] if vol_ma_3[i] > 0 else 0
            if (close[i] > r4_aligned[i] and 
                vol_ratio > 1.8 and 
                atr_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below S4 with volume confirmation and volatility filter
            elif (close[i] < s4_aligned[i] and 
                  vol_ratio > 1.8 and 
                  atr_aligned[i] > 0):
                position = -1
                signals[i] = -0.25
    
    return signals