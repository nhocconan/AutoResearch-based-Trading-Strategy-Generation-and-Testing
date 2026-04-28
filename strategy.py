#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from daily data (4-week lookback)
    # For each day, use the high/low/close of the previous 4 weeks (20 trading days)
    lookback = 20
    weekly_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    weekly_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    weekly_close = pd.Series(close_1d).rolling(window=lookback, min_periods=lookback).mean().values  # using mean as proxy for weekly close
    
    # Calculate pivot points and support/resistance levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    r4 = weekly_high + 3 * (pivot - weekly_low)
    s4 = weekly_low - 3 * (weekly_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 6-day EMA(34) for trend filter on daily timeframe
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike detector: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 34  # enough for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above EMA34 for long, below for short
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Volume spike filter
        vol_ok = vol_spike[i]
        
        # Fade at R3/S3: price touches extreme level and reverses
        fade_at_r3 = (high[i] >= r3_aligned[i]) and (close[i] < r3_aligned[i] * 0.999)  # touched R3 and closed below
        fade_at_s3 = (low[i] <= s3_aligned[i]) and (close[i] > s3_aligned[i] * 1.001)  # touched S3 and closed above
        
        # Breakout continuation at R4/S4: price breaks extreme level with volume
        breakout_at_r4 = (close[i] > r4_aligned[i]) and vol_ok
        breakout_at_s4 = (close[i] < s4_aligned[i]) and vol_ok
        
        # Long conditions: 
        # 1. Fade at S3 in uptrend OR
        # 2. Breakout at R4 in uptrend
        long_condition = (uptrend and fade_at_s3) or (uptrend and breakout_at_r4)
        
        # Short conditions: 
        # 1. Fade at R3 in downtrend OR
        # 2. Breakout at S4 in downtrend
        short_condition = (downtrend and fade_at_r3) or (downtrend and breakout_at_s4)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite signal or loss of trend
        elif position == 1 and (not uptrend or fade_at_r3):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not downtrend or fade_at_s3):
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

name = "6h_WeeklyPivot_FadeBreakout_EMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0