#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for context
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_w = df_w['close'].values
    ema_50_w = np.full(len(df_w), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(df_w)):
        if i == 0:
            ema_50_w[i] = close_w[i]
        elif np.isnan(ema_50_w[i-1]):
            ema_50_w[i] = close_w[i]
        else:
            ema_50_w[i] = alpha * close_w[i] + (1 - alpha) * ema_50_w[i-1]
    ema_50_w_aligned = align_htf_to_ltf(prices, df_w, ema_50_w)
    
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
    
    # Align daily values to daily timeframe (1:1 mapping)
    pp_aligned = align_htf_to_ltf(prices, df_d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_d, s4)
    prev_high_aligned = align_htf_to_ltf(prices, df_d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_d, prev_low)
    
    # Volume confirmation: 20-day average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(ema_50_w_aligned[i])):
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
            # Enter long: price closes above R4 with volume confirmation AND weekly uptrend
            if (close[i] > r4_aligned[i] and 
                volume[i] > vol_ma_20[i] * 1.5 and
                close[i] > ema_50_w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below S4 with volume confirmation AND weekly downtrend
            elif (close[i] < s4_aligned[i] and 
                  volume[i] > vol_ma_20[i] * 1.5 and
                  close[i] < ema_50_w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals