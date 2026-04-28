#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Calculate daily range for volatility filter
    daily_range = high_1d - low_1d
    avg_daily_range = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_uptrend = close_1w > ema34_1w
    weekly_downtrend = close_1w < ema34_1w
    
    # Align data to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    avg_range_aligned = align_htf_to_ltf(prices, df_1d, avg_daily_range)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Calculate 4h RSI for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(avg_range_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Get current weekly trend
        weekly_uptrend = weekly_uptrend_aligned[i] > 0.5
        weekly_downtrend = weekly_downtrend_aligned[i] > 0.5
        
        # Volatility filter: only trade when volatility is above average
        volatility_filter = daily_range[i] > avg_range_aligned[i] * 0.8
        
        # Fade conditions at S1/R1 with RSI extremes
        fade_s1 = close[i] <= s1_aligned[i] and rsi[i] < 30
        fade_r1 = close[i] >= r1_aligned[i] and rsi[i] > 70
        
        # Long conditions: fade at S1 in uptrend or ranging market
        long_condition = False
        if weekly_uptrend or (not weekly_uptrend and not weekly_downtrend):
            long_condition = fade_s1 and volatility_filter
        
        # Short conditions: fade at R1 in downtrend or ranging market
        short_condition = False
        if weekly_downtrend or (not weekly_uptrend and not weekly_downtrend):
            short_condition = fade_r1 and volatility_filter
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite signal or RSI neutral
        elif position == 1 and (close[i] >= pivot_aligned[i] or rsi[i] > 50):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] <= pivot_aligned[i] or rsi[i] < 50):
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

name = "4h_DailyPivot_Fade_S1R1_WeeklyTrend_VolFilter"
timeframe = "4h"
leverage = 1.0