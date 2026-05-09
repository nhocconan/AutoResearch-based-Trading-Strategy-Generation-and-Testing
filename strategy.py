#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian20_Breakout_1dVolume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    """
    1d Donchian breakout with 1d volume confirmation and 1w trend filter.
    - Long: Close > 20-day high + volume > 1.2x 20-day avg volume + weekly close > weekly SMA(10)
    - Short: Close < 20-day low + volume > 1.2x 20-day avg volume + weekly close < weekly SMA(10)
    - Exit: Opposite signal or price crosses weekly SMA(10)
    - Target: 10-20 trades/year on 1d timeframe
    """
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
    
    # 20-day Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-day average volume
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly SMA(10) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    sma10_1w = close_1w.rolling(window=10, min_periods=10).mean().values
    sma10_1w_aligned = align_htf_to_ltf(prices, df_1w, sma10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if weekly SMA data not ready
        if np.isnan(sma10_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition
        vol_condition = volume[i] > 1.2 * vol_avg_20[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume + weekly trend up
            if close[i] > high_20[i] and vol_condition and close[i] > sma10_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume + weekly trend down
            elif close[i] < low_20[i] and vol_condition and close[i] < sma10_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly SMA or opposite signal
            if close[i] < sma10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly SMA or opposite signal
            if close[i] > sma10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals