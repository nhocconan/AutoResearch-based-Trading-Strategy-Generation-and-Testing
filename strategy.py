#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot levels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h pivot points (classic)
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    r1_12h = 2 * pivot_12h - low_12h
    s1_12h = 2 * pivot_12h - high_12h
    r2_12h = pivot_12h + (high_12h - low_12h)
    s2_12h = pivot_12h - (high_12h - low_12h)
    r3_12h = high_12h + 2 * (pivot_12h - low_12h)
    s3_12h = low_12h - 2 * (high_12h - pivot_12h)
    r4_12h = pivot_12h + 3 * (high_12h - low_12h)
    s4_12h = pivot_12h - 3 * (high_12h - low_12h)
    
    # Align 12h pivots to 6h
    pivot_6h = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_6h = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_6h = align_htf_to_ltf(prices, df_12h, s1_12h)
    r2_6h = align_htf_to_ltf(prices, df_12h, r2_12h)
    s2_6h = align_htf_to_ltf(prices, df_12h, s2_12h)
    r3_6h = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_6h = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_6h = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_6h = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume moving average (20-period on 6h)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # need volume MA and EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(ema50_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume
            if close[i] > r4_6h[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S4 with volume
            elif close[i] < s4_6h[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price falls below pivot or R1
            if close[i] < pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above pivot or S1
            if close[i] > pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_Pivot_Exit"
timeframe = "6h"
leverage = 1.0