#/usr/bin/env python3
name = "4h_Donchian_20_Breakout_12hTrend_Volume"
timeframe = "4h"
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
    
    # Get 12h data for trend (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_12h = close_12h > ema50_12h
    trend_up_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_up_12h)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Calculate Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trend_up_12h_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(high_max_20[i]) or
            np.isnan(low_min_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + 12h uptrend + volume confirmation
            if (close[i] > high_max_20[i] and 
                trend_up_12h_aligned[i] and 
                volume[i] > 1.3 * vol_ma20_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low + 12h downtrend + volume confirmation
            elif (close[i] < low_min_20[i] and 
                  not trend_up_12h_aligned[i] and 
                  volume[i] > 1.3 * vol_ma20_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend changes
            if (close[i] < low_min_20[i] or 
                not trend_up_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend changes
            if (close[i] > high_max_20[i] or 
                trend_up_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals