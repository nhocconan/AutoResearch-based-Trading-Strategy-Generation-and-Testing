#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate weekly pivot points using weekly high/low/close
    # Get weekly data by taking every 5th day (approximation for 5 trading days)
    # We'll calculate weekly values using 5-day rolling window
    high_5d = pd.Series(high).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(low).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(close).rolling(window=5, min_periods=5).last().values
    
    # Calculate weekly pivot points (PP, R3, S3)
    pp_5d = (high_5d + low_5d + close_5d) / 3
    r3_5d = close_5d + (high_5d - low_5d) * 1.1  # R3 = Close + 1.1*(High-Low)
    s3_5d = close_5d - (high_5d - low_5d) * 1.1  # S3 = Close - 1.1*(High-Low)
    
    # Use previous period's values to avoid look-ahead
    pp_5d_prev = np.roll(pp_5d, 1)
    r3_5d_prev = np.roll(r3_5d, 1)
    s3_5d_prev = np.roll(s3_5d, 1)
    pp_5d_prev[0] = np.nan
    r3_5d_prev[0] = np.nan
    s3_5d_prev[0] = np.nan
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(pp_5d_prev[i]) or 
            np.isnan(r3_5d_prev[i]) or np.isnan(s3_5d_prev[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]
        
        if position == 0:
            # Long: Close breaks above R3 with volume spike and above 1d EMA trend
            if close[i] > r3_5d_prev[i] and vol_ok and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 with volume spike and below 1d EMA trend
            elif close[i] < s3_5d_prev[i] and vol_ok and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses back below S3 (mean reversion)
            if close[i] < s3_5d_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses back above R3
            if close[i] > r3_5d_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals