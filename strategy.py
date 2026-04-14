#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily 20-period high and low for Donchian channel
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily 10-period high and low for exit
    high_10_1d = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    low_10_1d = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Calculate median volume for volume spike filter
    vol_median = np.nanmedian(volume)
    
    # Create arrays for alignment
    ema_50_1d_arr = ema_50_1d
    high_20_1d_arr = high_20_1d
    low_20_1d_arr = low_20_1d
    high_10_1d_arr = high_10_1d
    low_10_1d_arr = low_10_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(60, n):
        # Get aligned daily data
        ema_50_1d_i = align_htf_to_ltf(prices, df_1d, ema_50_1d_arr)[i]
        high_20_1d_i = align_htf_to_ltf(prices, df_1d, high_20_1d_arr)[i]
        low_20_1d_i = align_htf_to_ltf(prices, df_1d, low_20_1d_arr)[i]
        high_10_1d_i = align_htf_to_ltf(prices, df_1d, high_10_1d_arr)[i]
        low_10_1d_i = align_htf_to_ltf(prices, df_1d, low_10_1d_arr)[i]
        
        if np.isnan(ema_50_1d_i) or \
           np.isnan(high_20_1d_i) or np.isnan(low_20_1d_i) or \
           np.isnan(high_10_1d_i) or np.isnan(low_10_1d_i):
            continue
        
        # Volume spike filter
        volume_spike = volume[i] > 2.0 * vol_median
        
        # Long entry: price breaks above daily Donchian high + volume spike + above EMA50
        if position == 0 and volume_spike:
            if close[i] > high_20_1d_i and close[i] > ema_50_1d_i:
                position = 1
                signals[i] = position_size
            elif close[i] < low_20_1d_i and close[i] < ema_50_1d_i:
                position = -1
                signals[i] = -position_size
        
        # Exit: price crosses 10-period opposite level
        elif position == 1 and close[i] < low_10_1d_i:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > high_10_1d_i:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_DailyDonchianBreakout_EMA50_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0