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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(21) for trend direction
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly EMA to daily (delayed for completed weekly bar)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Get daily data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channel on daily data
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily (delayed for completed daily bar)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate daily volume spike (volume > 2.0x 20-period average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # wait for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_aligned[i]) or
            np.isnan(low_20_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high with volume and above weekly EMA
            if close[i] > high_20_aligned[i-1] and volume_spike_aligned[i] and close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with volume and below weekly EMA
            elif close[i] < low_20_aligned[i-1] and volume_spike_aligned[i] and close[i] < ema_21_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-day low
            if close[i] < low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-day high
            if close[i] > high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA21_VolumeTrendFilter"
timeframe = "1d"
leverage = 1.0