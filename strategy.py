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
    
    # 1d data for context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_1d = np.zeros_like(close_1d)
    atr_1d[14:] = pd.Series(tr).rolling(window=14, min_periods=14).mean().values[13:]
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 6h Donchian(20) breakout
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    middle = (highest_high + lowest_low) / 2
    
    # Volume filter: volume > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    # Volatility filter: current 1d ATR > 0.8x median of last 50 bars
    atr_median = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=1).median()
    vol_filter = atr_1d_aligned > (0.8 * atr_median)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_threshold[i]) or np.isnan(vol_filter[i]):
            continue
        
        # Long: price breaks above Donchian high + volume + volatility
        if (close[i] > highest_high[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low + volume + volatility
        elif (close[i] < lowest_low[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle of channel (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < middle[i]) or
               (signals[i-1] == -0.25 and close[i] > middle[i]))):
            signals[i] = 0.0
        
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Donchian_Breakout_Vol_VolFilter"
timeframe = "6h"
leverage = 1.0