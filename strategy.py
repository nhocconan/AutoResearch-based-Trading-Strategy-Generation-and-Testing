#!/usr/bin/env python3
name = "12h_Donchian20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Donchian channel (20-period) on 12h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d volume spike (20-period average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (vol_ma_1d * 1.5)
    
    # Align indicators to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high}), high_20)
    low_20_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low}), low_20)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        if np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: close breaks above Donchian high + 1d uptrend + volume spike
            if close[i] > high_20_aligned[i] and close[i] > ema34_1d_aligned[i] and vol_spike_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: close breaks below Donchian low + 1d downtrend + volume spike
            elif close[i] < low_20_aligned[i] and close[i] < ema34_1d_aligned[i] and vol_spike_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close below Donchian low
            if close[i] < low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above Donchian high
            if close[i] > high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals