#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from 1d data (using prior week's OHLC)
    # We need to group into weeks and get the prior week's HLC
    # Since we don't have a direct weekly grouping, we'll use a rolling window approach
    # but for simplicity, we'll use the prior day's HLC as proxy for weekly pivot
    # Better approach: calculate weekly pivot from actual weekly data
    
    # Get actual weekly data for proper pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot points (standard formula)
    # Using prior week's HLC
    pp = (np.roll(high_1w, 1) + np.roll(low_1w, 1) + np.roll(close_1w, 1)) / 3
    r1 = 2 * pp - np.roll(low_1w, 1)
    s1 = 2 * pp - np.roll(high_1w, 1)
    r2 = pp + (np.roll(high_1w, 1) - np.roll(low_1w, 1))
    s2 = pp - (np.roll(high_1w, 1) - np.roll(low_1w, 1))
    r3 = high_1w + 2 * (pp - np.roll(low_1w, 1))
    s3 = low_1w - 2 * (np.roll(high_1w, 1) - pp)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike detector (volume > 1.5 * 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions at weekly pivot levels
        # Long breakout: price breaks above R2 with volume spike
        long_breakout = (close[i] > r2_aligned[i]) and vol_spike[i] and uptrend
        
        # Short breakdown: price breaks below S2 with volume spike
        short_breakdown = (close[i] < s2_aligned[i]) and vol_spike[i] and downtrend
        
        # Fade conditions at extreme levels (R3/S3)
        # Long fade: price rejects S3 with volume spike
        long_fade = (close[i] < s3_aligned[i] * 1.005) and (close[i] > s3_aligned[i] * 0.995) and vol_spike[i] and uptrend
        
        # Short fade: price rejects R3 with volume spike
        short_fade = (close[i] > r3_aligned[i] * 0.995) and (close[i] < r3_aligned[i] * 1.005) and vol_spike[i] and downtrend
        
        if (long_breakout or long_fade) and position <= 0:
            signals[i] = 0.25
            position = 1
        elif (short_breakdown or short_fade) and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: return to pivot point or opposite extreme
        elif position == 1 and (close[i] < pp_aligned[i] * 1.005 and close[i] > pp_aligned[i] * 0.995):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > pp_aligned[i] * 0.995 and close[i] < pp_aligned[i] * 1.005):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_BreakoutFade_VolumeSpike"
timeframe = "6h"
leverage = 1.0