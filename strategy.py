#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (S3, S2, S1, PP, R1, R2, R3)
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    # S2 = C - (H - L) * 1.1 / 6
    # R3 = C + (H - L) * 1.1 / 4
    # S3 = C - (H - L) * 1.1 / 4
    pivot_pp = np.full(len(close_1d), np.nan)
    r1 = np.full(len(close_1d), np.nan)
    s1 = np.full(len(close_1d), np.nan)
    r2 = np.full(len(close_1d), np.nan)
    s2 = np.full(len(close_1d), np.nan)
    r3 = np.full(len(close_1d), np.nan)
    s3 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        pp = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        range_hl = high_1d[i] - low_1d[i]
        pivot_pp[i] = pp
        r1[i] = close_1d[i] + range_hl * 1.1 / 12.0
        s1[i] = close_1d[i] - range_hl * 1.1 / 12.0
        r2[i] = close_1d[i] + range_hl * 1.1 / 6.0
        s2[i] = close_1d[i] - range_hl * 1.1 / 6.0
        r3[i] = close_1d[i] + range_hl * 1.1 / 4.0
        s3[i] = close_1d[i] - range_hl * 1.1 / 4.0
    
    # Align pivot levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pivot_pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current volume > 1.5x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below S1 or stoploss hit
            if (close[i] < s1_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above R1 or stoploss hit
            if (close[i] > r1_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries - mean reversion at S1/R1 with volume
            # Long: price touches S1 and reverses up, with volume
            if (close[i] <= s1_aligned[i] * 1.001 and  # Allow small tolerance
                close[i] > s1_aligned[i] * 0.999 and
                close[i] > open[i] and  # Bullish candle
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price touches R1 and reverses down, with volume
            elif (close[i] >= r1_aligned[i] * 0.999 and   # Allow small tolerance
                  close[i] <= r1_aligned[i] * 1.001 and
                  close[i] < open[i] and  # Bearish candle
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals