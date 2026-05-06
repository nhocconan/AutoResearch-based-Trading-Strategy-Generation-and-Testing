#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot + 1d trend filter + volume confirmation
# Uses daily Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout)
# Requires 1d EMA50 trend filter (price > EMA50 for long, < EMA50 for short)
# Volume confirmation (>1.5x 20-bar average) ensures participation
# Camarilla levels work well in ranging markets (fade at R3/S3) and trending (breakout at R4/S4)
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in both bull/bear: fades extremes in range, breaks with trend

name = "6h_Camarilla_R3S3_R4S4_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + (H - L) * 1.1 / 2
    # S3 = Pivot - (H - L) * 1.1 / 2
    # R4 = Pivot + (H - L) * 1.1
    # S4 = Pivot - (H - L) * 1.1
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3
    rang = high_1d[:-1] - low_1d[:-1]
    
    r3 = pivot + rang * 1.1 / 2
    s3 = pivot - rang * 1.1 / 2
    r4 = pivot + rang * 1.1
    s4 = pivot - rang * 1.1
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry conditions:
            # 1. Fade at S3: price crosses above S3 with close > S3
            # 2. Breakout at R4: price crosses above R4 with close > R4
            # Both require: price > EMA50 (uptrend) and volume confirmation
            fade_long = (close[i-1] <= s3_aligned[i] and close[i] > s3_aligned[i])
            breakout_long = (close[i-1] <= r4_aligned[i] and close[i] > r4_aligned[i])
            
            if ((fade_long or breakout_long) and 
                close[i] > ema_50_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            
            # Short entry conditions:
            # 1. Fade at R3: price crosses below R3 with close < R3
            # 2. Breakout at S4: price crosses below S4 with close < S4
            # Both require: price < EMA50 (downtrend) and volume confirmation
            fade_short = (close[i-1] >= r3_aligned[i] and close[i] < r3_aligned[i])
            breakout_short = (close[i-1] >= s4_aligned[i] and close[i] < s4_aligned[i])
            
            if ((fade_short or breakout_short) and 
                close[i] < ema_50_aligned[i] and 
                volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below S3 (fade level) or above R4 (breakout failure)
            if close[i] < s3_aligned[i] or close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above R3 (fade level) or below S4 (breakout failure)
            if close[i] > r3_aligned[i] or close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals