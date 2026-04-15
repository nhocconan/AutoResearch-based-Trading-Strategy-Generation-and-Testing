#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d VWAP filter and volume confirmation
# Long when price breaks above 20-bar high + above 1d VWAP + volume > 1.5x 20-bar median
# Short when price breaks below 20-bar low + below 1d VWAP + volume > 1.5x 20-bar median
# Exit when price returns to 20-bar midpoint or VWAP is crossed
# Designed for trend following in both bull and bear markets with strict entry to limit trades
# Conservative sizing (0.25) to balance return and risk

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_array = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_array)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above 20-bar high, above 1d VWAP, volume spike
        if (close[i] > high_20[i] and 
            close[i] > vwap_1d_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below 20-bar low, below 1d VWAP, volume spike
        elif (close[i] < low_20[i] and 
              close[i] < vwap_1d_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price returns to 20-bar midpoint or crosses VWAP
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] <= mid_20[i] or close[i] <= vwap_1d_aligned[i])) or
               (signals[i-1] == -0.25 and (close[i] >= mid_20[i] or close[i] >= vwap_1d_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_VWAP_Volume"
timeframe = "4h"
leverage = 1.0