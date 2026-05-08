#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = C + (H - L) * 1.1 / 2
    # S3 = C - (H - L) * 1.1 / 2
    # R4 = C + (H - L) * 1.1
    # S4 = C - (H - L) * 1.1
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.1 / 2
    s3_1d = close_1d - range_1d * 1.1 / 2
    r4_1d = close_1d + range_1d * 1.1
    s4_1d = close_1d - range_1d * 1.1
    
    # Align Camarilla levels to 6h timeframe (previous day's levels)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume spike and daily uptrend
            long_breakout = (close[i] > r4_1d_aligned[i] and 
                            ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                            volume_spike[i])
            
            # Short breakdown: price breaks below S4 with volume spike and daily downtrend
            short_breakdown = (close[i] < s4_1d_aligned[i] and 
                              ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                              volume_spike[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakdown:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below R3 (mean reversion)
            if close[i] < r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above S3 (mean reversion)
            if close[i] > s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals