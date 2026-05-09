#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyPivot_Direction"
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
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Standard pivot: P = (H + L + C)/3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # Resistance/Support levels
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_w, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_w, s3_w)
    
    # Get daily data for trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on daily close for trend filter
    close_d = df_d['close'].values
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_d_aligned = align_htf_to_ltf(prices, df_d, ema50_d)
    
    # Volume spike filter: current volume > 1.5 * 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # Need enough data for EMA50 (daily) and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema50_d_aligned[i]) or 
            np.isnan(pivot_w_aligned[i]) or
            np.isnan(r1_w_aligned[i]) or
            np.isnan(s1_w_aligned[i]) or
            np.isnan(r2_w_aligned[i]) or
            np.isnan(s2_w_aligned[i]) or
            np.isnan(r3_w_aligned[i]) or
            np.isnan(s3_w_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_d_val = ema50_d_aligned[i]
        pivot_w_val = pivot_w_aligned[i]
        r1_w_val = r1_w_aligned[i]
        s1_w_val = s1_w_aligned[i]
        r2_w_val = r2_w_aligned[i]
        s2_w_val = s2_w_aligned[i]
        r3_w_val = r3_w_aligned[i]
        s3_w_val = s3_w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Price above weekly pivot + daily uptrend + volume spike
            if close[i] > pivot_w_val and close[i] > ema50_d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below weekly pivot + daily downtrend + volume spike
            elif close[i] < pivot_w_val and close[i] < ema50_d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below weekly pivot or daily trend turns down
            if close[i] < pivot_w_val or close[i] < ema50_d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above weekly pivot or daily trend turns up
            if close[i] > pivot_w_val or close[i] > ema50_d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals