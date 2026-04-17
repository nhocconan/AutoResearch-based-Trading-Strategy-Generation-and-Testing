#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Breakout_Volume_TrendFilter_v2
Strategy: 4h Camarilla pivot (R1/S1) breakout with volume and trend filter.
Long: Price breaks above daily Camarilla R1 + volume > 1.5x 20-period avg + price > 4h EMA34
Short: Price breaks below daily Camarilla S1 + volume > 1.5x 20-period avg + price < 4h EMA34
Exit: Opposite Camarilla level break
Position size: 0.25
Uses daily pivot levels for structure, volume for confirmation, EMA34 for trend filter.
Designed to work in both bull and bear markets by requiring trend alignment.
Timeframe: 4h
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
    
    # Get daily data for Camarilla pivots
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    H_L = daily_high - daily_low
    camarilla_r1 = daily_close + H_L * 1.1 / 12
    camarilla_s1 = daily_close - H_L * 1.1 / 12
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s1)
    
    # Calculate 4h EMA34 for trend filter
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 4h volume average (20-period)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(34, n):  # warmup for EMA34
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below EMA34
        uptrend = close[i] > ema34[i]
        downtrend = close[i] < ema34[i]
        
        # Breakout signals
        breakout_up = close[i] > camarilla_r1_aligned[i]
        breakout_down = close[i] < camarilla_s1_aligned[i]
        
        if position == 0:
            # Long: Breakout above R1 + volume filter + uptrend
            if breakout_up and volume_filter and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 + volume filter + downtrend
            elif breakout_down and volume_filter and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Breakdown below S1 (opposite level)
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Breakout above R1 (opposite level)
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_Volume_TrendFilter_v2"
timeframe = "4h"
leverage = 1.0