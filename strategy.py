#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 1:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (standard)
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Pivot point: P = (H + L + C)/3
    pivot_d = (high_d + low_d + close_d) / 3.0
    # Calculate range
    range_d = high_d - low_d
    # Camarilla levels
    r3_d = close_d + range_d * 1.1 / 2
    s3_d = close_d - range_d * 1.1 / 2
    
    # Align daily Camarilla levels to 12h timeframe
    r3_d_aligned = align_htf_to_ltf(prices, df_d, r3_d)
    s3_d_aligned = align_htf_to_ltf(prices, df_d, s3_d)
    
    # Get weekly data for trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    # Calculate 5-period EMA on weekly close for trend
    close_w = df_w['close'].values
    ema_w = pd.Series(close_w).ewm(span=5, adjust=False, min_periods=5).mean().values
    ema_w_aligned = align_htf_to_ltf(prices, df_w, ema_w)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 5)  # Need enough data for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(r3_d_aligned[i]) or 
            np.isnan(s3_d_aligned[i]) or
            np.isnan(ema_w_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_d_aligned[i]
        s3_val = s3_d_aligned[i]
        ema_val = ema_w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price above R3 + price above weekly EMA + volume filter
            if close[i] > r3_val and close[i] > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below S3 + price below weekly EMA + volume filter
            elif close[i] < s3_val and close[i] < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below S3
            if close[i] < s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above R3
            if close[i] > r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals