#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_4hSMA50"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ===== 1D Trend Filter =====
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(34)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # ===== 4H SMA50 for Confluence =====
    close_s = pd.Series(close)
    sma50_4h = close_s.rolling(window=50, min_periods=50).mean().values
    
    # ===== Volume Spike Filter =====
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    # ===== Camarilla Pivot Levels (from previous 1D close) =====
    # Calculate pivot from previous day's data
    prev_close_1d = np.concatenate([[close_1d[0]], close_1d[:-1]])
    prev_high_1d = np.concatenate([[high_1d[0]], high_1d[:-1]])
    prev_low_1d = np.concatenate([[low_1d[0]], low_1d[:-1]])
    
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_hl = prev_high_1d - prev_low_1d
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(sma50_4h[i]) or
            np.isnan(vol_spike[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close > R1 + volume spike + price > SMA50 + 1d close > EMA34
            if (close[i] > r1_aligned[i] and 
                vol_spike[i] and 
                close[i] > sma50_4h[i] and 
                close_1d[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 + volume spike + price < SMA50 + 1d close < EMA34
            elif (close[i] < s1_aligned[i] and 
                  vol_spike[i] and 
                  close[i] < sma50_4h[i] and 
                  close_1d[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close < S1 OR volume spike reversal
            if close[i] < s1_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close > R1 OR volume spike reversal
            if close[i] > r1_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals