#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian channels identify strong breakout levels; breaks above/below the 20-period
# channel with volume confirmation indicate momentum. 1d EMA34 ensures alignment with
# daily trend to avoid false breakouts. Designed for 50-150 total trades over 4 years
# (12-37/year) on 12h timeframe. Works in bull/bear markets by taking breakouts only
# in direction of 1d EMA34.

name = "12h_Donchian20_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) from 1d data
    # Upper/lower channel based on prior day's high/low
    df_1d_for_channels = get_htf_data(prices, '1d')
    if len(df_1d_for_channels) < 20:
        return np.zeros(n)
    
    high_1d = df_1d_for_channels['high'].values
    low_1d = df_1d_for_channels['low'].values
    
    # 20-period rolling high/low on 1d data
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe (wait for prior day to complete)
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d_for_channels, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d_for_channels, lower_channel)
    
    # Volume confirmation: 2.0x 20-period average (20*12h = 10 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA34 and Donchian)
    start_idx = max(34, 20)  # 34 bars for EMA34, 20 bars for Donchian
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper channel with volume spike AND price > 1d EMA34 (bullish trend)
            if (close[i] > upper_channel_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower channel with volume spike AND price < 1d EMA34 (bearish trend)
            elif (close[i] < lower_channel_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below upper channel (failed breakout) OR price below 1d EMA34 (trend change)
            if close[i] < upper_channel_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above lower channel (failed breakdown) OR price above 1d EMA34 (trend change)
            if close[i] > lower_channel_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals