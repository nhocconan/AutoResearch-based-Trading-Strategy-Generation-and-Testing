#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily ATR(14) for volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR with Wilder's smoothing
    atr_1d = np.full_like(close_1d, np.nan)
    if len(tr) >= 14:
        atr_1d[13] = np.nanmean(tr[1:15])  # seed
        for i in range(14, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # === Weekly 20-period High/Low for breakout ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Highest high and lowest low over past 20 weeks
    highest_high_20w = np.full_like(high_1w, np.nan)
    lowest_low_20w = np.full_like(low_1w, np.nan)
    
    for i in range(len(high_1w)):
        if i >= 19:
            start_idx = i - 19
            highest_high_20w[i] = np.max(high_1w[start_idx:i+1])
            lowest_low_20w[i] = np.min(low_1w[start_idx:i+1])
        elif i > 0:
            highest_high_20w[i] = np.max(high_1w[0:i+1])
            lowest_low_20w[i] = np.min(low_1w[0:i+1])
        else:
            highest_high_20w[i] = high_1w[0]
            lowest_low_20w[i] = low_1w[0]
    
    # === Align indicators to daily timeframe ===
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    highest_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20w)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20w)
    
    # === Volume confirmation: today's volume > 1.5x 20-day average ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(highest_high_aligned[i]) or 
            np.isnan(lowest_low_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above 20-week high AND volatility filter AND volume confirmation
            if (close[i] > highest_high_aligned[i] and 
                atr_aligned[i] > 0 and  # volatility present
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below 20-week low AND volatility filter AND volume confirmation
            elif (close[i] < lowest_low_aligned[i] and 
                  atr_aligned[i] > 0 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or volatility collapse
        elif position == 1:
            # Exit long: price breaks below 20-week low OR volatility collapses
            if (close[i] < lowest_low_aligned[i] or 
                atr_aligned[i] < 0.5 * np.nanmedian(atr_aligned[max(0, i-49):i+1])):  # volatility drop
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 20-week high OR volatility collapses
            if (close[i] > highest_high_aligned[i] or 
                atr_aligned[i] < 0.5 * np.nanmedian(atr_aligned[max(0, i-49):i+1])):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_20WeekBreakout_VolATRFilter"
timeframe = "1d"
leverage = 1.0