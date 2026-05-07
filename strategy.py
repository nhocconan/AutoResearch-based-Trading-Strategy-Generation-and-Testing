# 4h_Donchian_Volume_Trend_v1
# Hypothesis: Donchian(20) breakout on 4h with volume confirmation and 1d EMA50 trend filter.
# This strategy captures breakouts with institutional volume while avoiding false signals
# through higher timeframe trend alignment. Designed for 15-30 trades/year to minimize
# fee drag while maintaining edge in both bull and bear markets through trend filtering.

name = "4h_Donchian_Volume_Trend_v1"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Volume confirmation
            vol_ok = volume[i] > vol_ma[i]
            
            # Long: price breaks above Donchian high + above 1d EMA50 + volume
            if (close[i] > high_max[i] and 
                close[i] > ema_50_1d_aligned[i] and vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below 1d EMA50 + volume
            elif (close[i] < low_min[i] and 
                  close[i] < ema_50_1d_aligned[i] and vol_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or trend changes
            if (close[i] < low_min[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or trend changes
            if (close[i] > high_max[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals