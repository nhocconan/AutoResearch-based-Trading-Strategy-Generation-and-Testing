#!/usr/bin/env python3
name = "6h_Donchian20_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d data for trend and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d volume spike filter (20-period average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (vol_ma_1d * 1.5)
    
    # Align 1d indicators to 6h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 6h Donchian channels (20-period)
    high_max = np.zeros(n)
    low_min = np.zeros(n)
    for i in range(n):
        if i < 19:
            high_max[i] = np.nan
            low_min[i] = np.nan
        else:
            high_max[i] = np.max(high[i-19:i+1])
            low_min[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper + 1d uptrend + volume spike
            if close[i] > high_max[i] and close[i] > ema34_1d_aligned[i] and vol_spike_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + 1d downtrend + volume spike
            elif close[i] < low_min[i] and close[i] < ema34_1d_aligned[i] and vol_spike_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower
            if close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper
            if close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals