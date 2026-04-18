#!/usr/bin/env python3
"""
12h Pivot Point Reversal + Volume Spike + 1d Trend Filter
Hypothesis: At key pivot levels (R1/S1), price often reverses. Combined with volume spike (institutional interest) and 1d trend filter (price vs EMA34), it captures high-probability reversals in both bull and bear markets. Low trade frequency due to strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def calculate_ema(data, span):
    """Calculate EMA with proper handling of NaN"""
    result = np.full_like(data, np.nan, dtype=float)
    if len(data) == 0:
        return result
    alpha = 2 / (span + 1)
    result[0] = data[0]
    for i in range(1, len(data)):
        if np.isnan(data[i]):
            result[i] = result[i-1]
        else:
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P, R1, S1, R2, S2"""
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    return p, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot points (our trading timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate pivot points on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot points for each 12h bar
    p_vals, r1_vals, s1_vals, r2_vals, s2_vals = calculate_pivot_points(high_12h, low_12h, close_12h)
    
    # Align pivot levels to lower timeframe (15m equivalent, but we'll use close alignment)
    p_aligned = align_ltf_to_htf(prices, df_12h, p_vals)
    r1_aligned = align_ltf_to_htf(prices, df_12h, r1_vals)
    s1_aligned = align_ltf_to_htf(prices, df_12h, s1_vals)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = calculate_ema(close_1d, 34)
    ema_34_1d_aligned = align_ltf_to_htf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.5x 24-period average (24*12h = 12 days)
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 24:
            vol_ma[i] = np.mean(volume[max(0, i-23):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-23:i+1])
    vol_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(p_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        p_val = p_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_34 = ema_34_1d_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: price at or below S1 with bullish bias (price > EMA34) + volume spike
            if (close[i] <= s1_val * 1.002 and  # Allow small buffer
                close[i] > ema_34 and
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price at or above R1 with bearish bias (price < EMA34) + volume spike
            elif (close[i] >= r1_val * 0.998 and  # Allow small buffer
                  close[i] < ema_34 and
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above R1 or trend turns bearish
            if close[i] >= r1_val or close[i] < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below S1 or trend turns bullish
            if close[i] <= s1_val or close[i] > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_PivotPoint_Reversal_VolumeSpike_1dTrendFilter"
timeframe = "12h"
leverage = 1.0