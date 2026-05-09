#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    pivot_prev = (high_prev + low_prev + close_prev) / 3.0
    r3 = pivot_prev + (range_prev * 1.1 / 4)
    s3 = pivot_prev - (range_prev * 1.1 / 4)
    r4 = pivot_prev + (range_prev * 1.1 / 2)
    s4 = pivot_prev - (range_prev * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    pivot_prev_aligned = align_htf_to_ltf(prices, df_1d, pivot_prev)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 1d trend filter (EMA50)
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(pivot_prev_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_prev_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        ema_50_val = ema_50_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price breaks above R3 + above 1d EMA50 + volume filter
            if close[i] > r3_val and close[i] > ema_50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below S3 + below 1d EMA50 + volume filter
            elif close[i] < s3_val and close[i] < ema_50_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below R3 or below 1d EMA50
            if close[i] < r3_val or close[i] < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above S3 or above 1d EMA50
            if close[i] > s3_val or close[i] > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals