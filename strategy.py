#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_Pivot_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1-day Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot: P = (H + L + C)/3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels (resistance/support)
    r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 2
    r2_1d = close_1d + (high_1d - low_1d) * 1.1 / 4
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 6
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 6
    s2_1d = close_1d - (high_1d - low_1d) * 1.1 / 4
    s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align 1d Camarilla levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Get 1d data for trend filter (EMA50)
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or
            np.isnan(s2_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_1d_aligned[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        r2_val = r2_1d_aligned[i]
        s2_val = s2_1d_aligned[i]
        r3_val = r3_1d_aligned[i]
        s3_val = s3_1d_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price breaks above R1 + above 1d EMA50 + volume filter
            if close[i] > r1_val and close[i] > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below S1 + below 1d EMA50 + volume filter
            elif close[i] < s1_val and close[i] < ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below R2 or below pivot
            if close[i] < r2_val or close[i] < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above S2 or above pivot
            if close[i] > s2_val or close[i] > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals