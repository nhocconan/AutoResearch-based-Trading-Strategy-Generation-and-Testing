#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla pivot levels (standard formula)
    # Using previous day's H/L/C to calculate current day's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d_prev) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels (standard multipliers)
    # Resistance levels
    r1_1d = close_1d_prev + (range_1d * 1.1 / 12)
    r2_1d = close_1d_prev + (range_1d * 1.1 / 6)
    r3_1d = close_1d_prev + (range_1d * 1.1 / 4)
    r4_1d = close_1d_prev + (range_1d * 1.1 / 2)
    # Support levels
    s1_1d = close_1d_prev - (range_1d * 1.1 / 12)
    s2_1d = close_1d_prev - (range_1d * 1.1 / 6)
    s3_1d = close_1d_prev - (range_1d * 1.1 / 4)
    s4_1d = close_1d_prev - (range_1d * 1.1 / 2)
    
    # Align daily Camarilla levels to 4h timeframe (wait for daily bar to close)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)  # Avoid division by zero
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price above S1 + above 1d EMA34 + volume confirmation
            if (close[i] > s1_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and
                vol_ratio[i] > 1.5):
                # Check if we're not too far above R1 (avoid chasing extreme extension)
                if close[i] <= r1_1d_aligned[i] * 1.02:  # Within 2% above R1
                    signals[i] = 0.25
                    position = 1
            # Short conditions: price below R1 + below 1d EMA34 + volume confirmation
            elif (close[i] < r1_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  vol_ratio[i] > 1.5):
                # Check if we're not too far below S1 (avoid chasing extreme extension)
                if close[i] >= s1_1d_aligned[i] * 0.98:  # Within 2% below S1
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below S1 OR below 1d EMA34
            if close[i] < s1_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R1 OR above 1d EMA34
            if close[i] > r1_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals