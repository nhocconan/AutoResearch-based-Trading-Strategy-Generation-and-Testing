#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyCamarilla_S1_S4_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla pivot calculation
    pivot = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    s1 = close_1w - (range_1w * 1.0 / 6)
    s2 = close_1w - (range_1w * 2.0 / 6)
    s3 = close_1w - (range_1w * 3.0 / 6)
    s4 = close_1w - (range_1w * 4.0 / 6)
    r1 = close_1w + (range_1w * 1.0 / 6)
    r2 = close_1w + (range_1w * 2.0 / 6)
    r3 = close_1w + (range_1w * 3.0 / 6)
    r4 = close_1w + (range_1w * 4.0 / 6)
    
    # Align weekly Camarilla levels to daily timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    
    # Daily EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(ema50[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1, price above EMA50, volume spike
            long_cond = (close[i] > r1_aligned[i] and 
                        close[i] > ema50[i] and
                        volume_spike[i])
            
            # Short: Price breaks below S1, price below EMA50, volume spike
            short_cond = (close[i] < s1_aligned[i] and 
                         close[i] < ema50[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below S4 OR price crosses below EMA50
            if close[i] < s4_aligned[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above R4 OR price crosses above EMA50
            if close[i] > r4_aligned[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals