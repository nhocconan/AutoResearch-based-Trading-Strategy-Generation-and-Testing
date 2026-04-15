#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily ATR for volatility filter (14-period)
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    tr1 = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1]))
    tr2 = np.maximum(np.abs(low_d[1:] - close_d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # 12h Donchian channel (20-period)
    data_12h = get_htf_data(prices, '12h')
    high_12h = data_12h['high'].values
    low_12h = data_12h['low'].values
    high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    high_20_12h_aligned = align_htf_to_ltf(prices, data_12h, high_20_12h)
    low_20_12h_aligned = align_htf_to_ltf(prices, data_12h, low_20_12h)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20_12h_aligned[i]) or np.isnan(low_20_12h_aligned[i]) or
            np.isnan(atr_14d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Volatility filter: avoid extremes (0.5x to 3.0x of median ATR)
        atr_median = pd.Series(atr_14d_aligned).rolling(window=50, min_periods=50).median()
        vol_filter = (atr_14d_aligned[i] > 0.5 * atr_median[i]) and (atr_14d_aligned[i] < 3.0 * atr_median[i])
        
        # Long: 12h Donchian breakout up + volume spike + volatility filter
        if (close[i] > high_20_12h_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: 12h Donchian breakout down + volume spike + volatility filter
        elif (close[i] < low_20_12h_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price re-enters 12h Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < high_20_12h_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > low_20_12h_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian20_Vol1.5x_ATR14dFilter"
timeframe = "12h"
leverage = 1.0