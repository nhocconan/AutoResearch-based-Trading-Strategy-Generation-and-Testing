#!/usr/bin/env python3
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
    
    # Daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly pivot calculation using daily data
    # We'll use the previous week's data to calculate pivot for current week
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    # Pivot point calculation (standard formula)
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot_point - weekly_low
    s1 = 2 * pivot_point - weekly_high
    r2 = pivot_point + (weekly_high - weekly_low)
    s2 = pivot_point - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot_point - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot_point)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_point_6h = align_htf_to_ltf(prices, df_1d, pivot_point)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # 60-period EMA for trend filter (6h timeframe)
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Volume confirmation: current > 1.5x median of last 30 periods
    vol_median = pd.Series(volume).rolling(window=30, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_point_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(ema_60[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long conditions:
        # 1. Price above 60-period EMA (uptrend filter)
        # 2. Price breaks above R3 level with volume confirmation
        # 3. Strong bullish momentum
        if (close[i] > ema_60[i] and 
            close[i] > r3_6h[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short conditions:
        # 1. Price below 60-period EMA (downtrend filter)
        # 2. Price breaks below S3 level with volume confirmation
        # 3. Strong bearish momentum
        elif (close[i] < ema_60[i] and 
              close[i] < s3_6h[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit conditions:
        # Long exit: price falls back below pivot point
        # Short exit: price rises back above pivot point
        elif i > 0:
            if signals[i-1] == 0.25 and close[i] < pivot_point_6h[i]:
                signals[i] = 0.0
            elif signals[i-1] == -0.25 and close[i] > pivot_point_6h[i]:
                signals[i] = 0.0
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_R3S3_Breakout"
timeframe = "6h"
leverage = 1.0