#!/usr/bin/env python3
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
    
    # Get 12h data for weekly pivot points
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 3:
        return np.zeros(n)
    
    # Calculate weekly pivot points from 12h data (3 bars = ~1.5 days, use lookback of 3)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Use last 3 periods to calculate pivot (weekly approximation)
    pivot = np.full(len(close_12h), np.nan)
    r1 = np.full(len(close_12h), np.nan)
    s1 = np.full(len(close_12h), np.nan)
    r2 = np.full(len(close_12h), np.nan)
    s2 = np.full(len(close_12h), np.nan)
    r3 = np.full(len(close_12h), np.nan)
    s3 = np.full(len(close_12h), np.nan)
    r4 = np.full(len(close_12h), np.nan)
    s4 = np.full(len(close_12h), np.nan)
    
    for i in range(2, len(close_12h)):
        # Use previous 3 periods for pivot calculation
        phigh = np.max(high_12h[i-2:i+1])
        plow = np.min(low_12h[i-2:i+1])
        pclose = close_12h[i]
        pivot[i] = (phigh + plow + pclose) / 3.0
        r1[i] = 2 * pivot[i] - plow
        s1[i] = 2 * pivot[i] - phigh
        r2[i] = pivot[i] + (phigh - plow)
        s2[i] = pivot[i] - (phigh - plow)
        r3[i] = phigh + 2 * (pivot[i] - plow)
        s3[i] = plow - 2 * (phigh - pivot[i])
        r4[i] = r3[i] + (r2[i] - r1[i])
        s4[i] = s3[i] - (s2[i] - s1[i])
    
    # Align pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_12h, pivot)
    r1_6h = align_htf_to_ltf(prices, df_12h, r1)
    s1_6h = align_htf_to_ltf(prices, df_12h, s1)
    r2_6h = align_htf_to_ltf(prices, df_12h, r2)
    s2_6h = align_htf_to_ltf(prices, df_12h, s2)
    r3_6h = align_htf_to_ltf(prices, df_12h, r3)
    s3_6h = align_htf_to_ltf(prices, df_12h, s3)
    r4_6h = align_htf_to_ltf(prices, df_12h, r4)
    s4_6h = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume confirmation - 20-period average
    vol_ma20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ma20_values = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ma20[:] = vol_ma20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_filter = volume[i] > vol_ma20[i] * 1.5
        
        # Fade at R3/S3, breakout continuation at R4/S4
        long_setup = (close[i] <= s3_6h[i] and close[i] > s4_6h[i]) and vol_filter  # bounce from S3, above S4
        short_setup = (close[i] >= r3_6h[i] and close[i] < r4_6h[i]) and vol_filter  # reject at R3, below R4
        long_breakout = close[i] > r4_6h[i] and vol_filter  # break above R4
        short_breakout = close[i] < s4_6h[i] and vol_filter  # break below S4
        
        if (long_setup or long_breakout) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (short_setup or short_breakout) and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (close[i] < pivot_6h[i] or (close[i] < s1_6h[i] and vol_filter)):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > pivot_6h[i] or (close[i] > r1_6h[i] and vol_filter)):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_pivot_r3s3_r4s4_vol_filter_v1"
timeframe = "6h"
leverage = 1.0