#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Donchian channel breakouts capture momentum. Weekly pivot direction (from 1w timeframe) 
provides structural bias to avoid counter-trend trades. Volume confirmation ensures participation. 
Designed for 6h timeframe to achieve 50-150 total trades over 4 years. Works in bull markets 
(breakouts above upper band in up bias) and bear markets (breakouts below lower band in down bias).
"""

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
    
    # Get 1d data for EMA trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot direction (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    # Weekly pivot direction: bullish if close > P, bearish if close < P
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_bias = np.where(weekly_close > weekly_pivot, 1,  # bullish
                          np.where(weekly_close < weekly_pivot, -1, 0))  # bearish/neutral
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Calculate Donchian(20) channels on 6h data
    # Upper band = highest high of last 20 periods, Lower band = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20), EMA, volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(weekly_bias_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        bias = weekly_bias_aligned[i]
        vol_spike = volume_spike[i]
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian band AND weekly bias bullish AND volume spike AND price > EMA
            long_entry = (curr_high > upper_band) and (bias == 1) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below lower Donchian band AND weekly bias bearish AND volume spike AND price < EMA
            short_entry = (curr_low < lower_band) and (bias == -1) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below lower Donchian band OR price crosses below EMA (trend change)
            if (curr_low < lower_band) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above upper Donchian band OR price crosses above EMA (trend change)
            if (curr_high > upper_band) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeSpike"
timeframe = "6h"
leverage = 1.0