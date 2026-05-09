#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
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
    
    # Get daily data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla pivot levels (standard formula)
    # Using previous day's data (already complete when we get it)
    prev_day_high = df_1d['high'].shift(1).values    # Previous day high
    prev_day_low = df_1d['low'].shift(1).values      # Previous day low
    prev_day_close = df_1d['close'].shift(1).values  # Previous day close
    
    # Calculate pivot point and support/resistance levels
    pivot_point = (prev_day_high + prev_day_low + prev_day_close) / 3.0
    range_ = prev_day_high - prev_day_low
    r1 = pivot_point + (range_ * 1.0 / 8.0)
    s1 = pivot_point - (range_ * 1.0 / 8.0)
    r2 = pivot_point + (range_ * 2.0 / 8.0)
    s2 = pivot_point - (range_ * 2.0 / 8.0)
    r3 = pivot_point + (range_ * 3.0 / 8.0)
    s3 = pivot_point - (range_ * 3.0 / 8.0)
    r4 = pivot_point + (range_ * 4.0 / 8.0)
    s4 = pivot_point - (range_ * 4.0 / 8.0)
    
    # Align daily pivot levels to 4h timeframe
    pivot_point_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 34-period EMA on daily close for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need 34 for daily EMA and 20 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(pivot_point_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot = pivot_point_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        r4_level = r4_aligned[i]
        s4_level = s4_aligned[i]
        ema_1d = ema_34_1d_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Price breaks above R1 with volume AND price > daily EMA34 (uptrend)
            if close[i] > r1_level and vol > 1.5 * vol_ma_val and close[i] > ema_1d:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below S1 with volume AND price < daily EMA34 (downtrend)
            elif close[i] < s1_level and vol > 1.5 * vol_ma_val and close[i] < ema_1d:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below S1 OR trend reverses (price < daily EMA34)
            if close[i] < s1_level or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above R1 OR trend reverses (price > daily EMA34)
            if close[i] > r1_level or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals