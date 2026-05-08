#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1d average volume for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 1d Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get previous day's data
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift[0] = np.nan
    low_1d_shift[0] = np.nan
    close_1d_shift[0] = np.nan
    
    pivot = (high_1d_shift + low_1d_shift + close_1d_shift) / 3
    range_ = high_1d_shift - low_1d_shift
    r1 = close_1d_shift + (range_ * 1.1 / 12)
    s1 = close_1d_shift - (range_ * 1.1 / 12)
    r3 = close_1d_shift + (range_ * 1.1 / 4)
    s3 = close_1d_shift - (range_ * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current volume > 2.0 * 20-day average volume
    volume_spike = volume > (2.0 * vol_avg_1d_aligned)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_4h_val = ema50_4h_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above S1 + uptrend (4h EMA50) + volume spike
            if (close[i] > s1_val and 
                close[i] > ema50_4h_val and 
                vol_spike):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below R1 + downtrend + volume spike
            elif (close[i] < r1_val and 
                  close[i] < ema50_4h_val and 
                  vol_spike):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR trend turns down (below 4h EMA50)
            if (close[i] < s3_val or close[i] < ema50_4h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R3 OR trend turns up (above 4h EMA50)
            if (close[i] > r3_val or close[i] > ema50_4h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals