#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla Pivot Point S3/R3 breakout with 12h trend filter and volume confirmation.
Long when price breaks above R3 with bullish 12h trend and volume spike.
Short when price breaks below S3 with bearish 12h trend and volume spike.
Exit when price returns to pivot point (P).
Uses 12h EMA50 for trend filter to capture medium-term trend and avoid whipsaws.
Designed for low trade frequency (20-40/year) to minimize fee drift.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter and pivot calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 55:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate daily OHLC for Camarilla pivot levels
    high_d = df_12h['high'].values
    low_d = df_12h['low'].values
    close_d = df_12h['close'].values
    
    # Previous day's OHLC for today's pivot levels
    # Shift by 1 to use previous day's data
    high_prev = np.roll(high_d, 1)
    low_prev = np.roll(low_d, 1)
    close_prev = np.roll(close_d, 1)
    # First day has no previous, set to NaN
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_val = high_prev - low_prev
    
    # Camarilla levels (S3 and R3)
    s3 = close_prev - (range_val * 1.1 / 4)
    r3 = close_prev + (range_val * 1.1 / 4)
    
    # Align all levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA lookback
        # Skip if data not ready
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with bullish 12h trend and volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema50_aligned[i] and  # Bullish trend: price above EMA50
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with bearish 12h trend and volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema50_aligned[i] and  # Bearish trend: price below EMA50
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to pivot point
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to pivot
                if close[i] <= pivot_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to pivot
                if close[i] >= pivot_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_S3R3_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0
#%%