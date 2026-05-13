#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1DTrend_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE for Camarilla pivot, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Based on previous day's OHLC
    pp = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    r1 = pp + (high_1d[:-1] - low_1d[:-1]) * 1.0 / 8.0
    s1 = pp - (high_1d[:-1] - low_1d[:-1]) * 1.0 / 8.0
    
    # Prepend NaN for first day (no previous day)
    r1 = np.concatenate([[np.nan], r1])
    s1 = np.concatenate([[np.nan], s1])
    
    # Calculate 1D EMA34 for trend filter
    close_s_1d = pd.Series(close_1d)
    ema34_1d = close_s_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1D volume average (20-period) for volume spike
    volume_s_1d = pd.Series(volume_1d)
    vol_avg_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align all 1D indicators to 4H timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 4H close for breakout and current volume
    volume_s = pd.Series(volume)
    vol_avg_4h = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(vol_avg_4h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA34
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume spike: current 4H volume > 1.5x average 1D volume (scaled)
        vol_spike = volume[i] > (vol_avg_1d_aligned[i] * 1.5)
        
        if position == 0:
            # LONG: Break above R1 in uptrend with volume spike
            if uptrend and close[i] > r1_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 in downtrend with volume spike
            elif downtrend and close[i] < s1_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns below R1 or trend changes
            if close[i] < r1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns above S1 or trend changes
            if close[i] > s1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals