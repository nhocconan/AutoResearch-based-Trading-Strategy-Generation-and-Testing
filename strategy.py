#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # where C = (H+L+CLOSE)/3 of previous day
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # handle first value
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    pivot_point = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_hl = prev_high_1d - prev_low_1d
    r3 = pivot_point + (range_hl * 1.1 / 4.0)
    s3 = pivot_point - (range_hl * 1.1 / 4.0)
    r4 = pivot_point + (range_hl * 1.1 / 2.0)
    s4 = pivot_point - (range_hl * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike: current volume > 2.0x 24-period average (4 days)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price above R4 with volume spike and daily uptrend
            long_breakout = (close[i] > r4_aligned[i] and 
                            volume_spike[i] and
                            ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1])
            
            # Short breakdown: price below S4 with volume spike and daily downtrend
            short_breakdown = (close[i] < s4_aligned[i] and 
                              volume_spike[i] and
                              ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakdown:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below S3 (mean reversion level)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R3 (mean reversion level)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals