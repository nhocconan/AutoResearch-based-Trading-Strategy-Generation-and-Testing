#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day Donchian breakout and volume confirmation.
# Uses 1-day Donchian breakout for trend direction, 4-hour breakout for entry timing,
# and volume filter to confirm momentum. Designed for 20-30 trades/year to avoid
# fee drag while capturing major moves in both bull and bear markets.
name = "4h_1d_Donchian20_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Donchian20 trend (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Donchian channels: 20-period high/low
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, high_20_1d)
    low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, low_20_1d)
    
    # Get 4h data for Donchian10 breakout (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    # Donchian channels: 10-period high/low
    high_10_4h = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    low_10_4h = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    high_10_4h_aligned = align_htf_to_ltf(prices, df_4h, high_10_4h)
    low_10_4h_aligned = align_htf_to_ltf(prices, df_4h, low_10_4h)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(high_20_1d_aligned[i]) or np.isnan(low_20_1d_aligned[i]) or 
            np.isnan(high_10_4h_aligned[i]) or np.isnan(low_10_4h_aligned[i]) or 
            np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1d Donchian high AND breaks 4h Donchian high with volume
            if (close[i] > high_20_1d_aligned[i] and 
                close[i] > high_10_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d Donchian low AND breaks 4h Donchian low with volume
            elif (close[i] < low_20_1d_aligned[i] and 
                  close[i] < low_10_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1d Donchian low or 4h Donchian low
            if close[i] < low_20_1d_aligned[i] or close[i] < low_10_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1d Donchian high or 4h Donchian high
            if close[i] > high_20_1d_aligned[i] or close[i] > high_10_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals