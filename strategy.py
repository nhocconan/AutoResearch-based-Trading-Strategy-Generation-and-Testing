#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout direction with 1d volume confirmation and session filter
# Uses 4h Donchian(20) for trend direction: long when price > 4h Donchian upper, short when price < 4h Donchian lower
# Confirmed by 1d volume spike (volume > 1.5 * 20-period average) to avoid false breakouts
# Only trades during 08-20 UTC session to reduce noise and whipsaws
# Discrete position sizing 0.20 to limit trades to ~15-37/year and minimize fee drag
# Works in bull/bear markets: follows institutional breakout direction with volume validation

name = "1h_4h_1d_donchian_breakout_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian(20) channels
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_upper_4h = rolling_max(high_4h, 20)
    donchian_lower_4h = rolling_min(low_4h, 20)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume average (20-period)
    def rolling_mean(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.mean(arr[i-window+1:i+1])
        return res
    
    volume_ma_1d = rolling_mean(volume_1d, 20)
    volume_spike_1d = np.where(volume_ma_1d > 0, volume_1d / volume_ma_1d, 1.0)
    
    # Align 4h Donchian channels to 1h timeframe
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Align 1d volume spike to 1h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price breaks below Donchian lower or volume confirmation fails
            if close[i] <= donchian_lower_4h_aligned[i] or volume_spike_1d_aligned[i] < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit short if price breaks above Donchian upper or volume confirmation fails
            if close[i] >= donchian_upper_4h_aligned[i] or volume_spike_1d_aligned[i] < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price breaks above Donchian upper with volume confirmation
            if close[i] > donchian_upper_4h_aligned[i] and volume_spike_1d_aligned[i] >= 1.5:
                position = 1
                signals[i] = 0.20
            # Enter short: price breaks below Donchian lower with volume confirmation
            elif close[i] < donchian_lower_4h_aligned[i] and volume_spike_1d_aligned[i] >= 1.5:
                position = -1
                signals[i] = -0.20
    
    return signals