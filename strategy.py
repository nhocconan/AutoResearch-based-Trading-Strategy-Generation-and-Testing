#!/usr/bin/env python3
"""
Hypothesis: 6-hour Elder Ray Power with Weekly Trend Filter and Volume Confirmation.
Uses daily Bull Power (Close - EMA13) and Bear Power (EMA13 - High) to measure buying/selling pressure.
Trades in direction of weekly trend (EMA34) when power exceeds volume-adjusted threshold.
Designed to work in both bull and bear markets by using weekly trend as filter.
Target: 15-35 trades/year per symbol (60-140 total over 4 years) to minimize fee drag.
"""
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
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (daily)
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power (daily)
    bull_power_1d = close_1d - ema13_1d  # Close - EMA13
    bear_power_1d = ema13_1d - df_1d['high'].values  # EMA13 - High
    
    # Align Elder Ray powers to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 for weekly trend
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get 6h data for volume filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h volume MA(20)
    vol_6h = df_6h['volume'].values
    vol_ma_20_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Elder Ray, volume MA, and weekly EMA
    start_idx = max(13, 20, 34)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_20_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        trend_1w = ema34_1w_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_6h_aligned[i]
        
        # Volume filter: volume > 1.3x 6h average
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Power threshold: 0.5% of price for significance
        power_threshold = close[i] * 0.005
        
        # Entry conditions
        if position == 0:
            # Long: Bull power > threshold + volume + weekly uptrend
            if bull_power > power_threshold and vol_filter and close[i] > trend_1w:
                signals[i] = size
                position = 1
            # Short: Bear power > threshold + volume + weekly downtrend
            elif bear_power > power_threshold and vol_filter and close[i] < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bull power turns negative or weekly trend turns down
            if bull_power <= 0 or close[i] < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bear power turns negative or weekly trend turns up
            if bear_power <= 0 or close[i] > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRayPower_WeeklyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0