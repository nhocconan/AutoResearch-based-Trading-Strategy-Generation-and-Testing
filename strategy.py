#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 6h data
    # Highest high of last 20 periods
    high_series = pd.Series(high)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods
    low_series = pd.Series(low)
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: EMA50 on weekly close
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume filter: current volume > 1.8 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = df_1d['volume'].values > (vol_ma * 1.8)
    
    # Align weekly EMA and daily volume to 6h
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 50)  # Need enough data for Donchian and weekly EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = donchian_high[i]
        lower = donchian_low[i]
        weekly_trend = ema50_1w_aligned[i]
        vol_ok = volume_filter_aligned[i]
        
        if position == 0:
            # Enter long: break above upper Donchian with volume and above weekly trend
            if close[i] > upper and close[i] > weekly_trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower Donchian with volume and below weekly trend
            elif close[i] < lower and close[i] < weekly_trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below midpoint of Donchian channel
            midpoint = (upper + lower) * 0.5
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above midpoint of Donchian channel
            midpoint = (upper + lower) * 0.5
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals