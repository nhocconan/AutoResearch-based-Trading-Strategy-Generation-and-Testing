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
    
    # Get 1d data for trend filter and Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d Donchian channels (20-day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: require volume > 2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 50  # need 50 for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian channel in uptrend with volume confirmation
            if (close[i] > upper_channel_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel in downtrend with volume confirmation
            elif (close[i] < lower_channel_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price closes below lower Donchian channel
            if close[i] < lower_channel_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above upper Donchian channel
            if close[i] > upper_channel_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_20_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0