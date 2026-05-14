#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_DonchianBreakout_WeeklyTrend_VolumeFilter"
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
    
    # Get 12h data for Donchian channels (20 periods)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 1d data for weekly trend filter (using 5-day EMA as proxy for weekly trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 1d EMA5 for trend filter
    ema5_1d = pd.Series(df_1d['close']).ewm(span=5, adjust=False, min_periods=5).mean().values
    ema5_1d_aligned = align_htf_to_ltf(prices, df_1d, ema5_1d)
    
    # Volume filter: current 6h volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)  # Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema5_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        trend = ema5_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: break above Donchian high + above 1d EMA5 trend + volume filter
            if close[i] > upper and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below Donchian low + below 1d EMA5 trend + volume filter
            elif close[i] < lower and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below Donchian low
            if close[i] < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above Donchian high
            if close[i] > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals