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
    
    # 1d Donchian channel (breakout structure)
    high_1d = get_htf_data(prices, '1d')
    high_20 = pd.Series(high_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(high_1d['low']).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, high_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, high_1d, low_20)
    
    # 1d ATR for volatility filter
    high_1d_arr = high_1d['high'].values
    low_1d_arr = high_1d['low'].values
    close_1d_arr = high_1d['close'].values
    tr1 = np.maximum(high_1d_arr[1:] - low_1d_arr[1:], np.abs(high_1d_arr[1:] - close_1d_arr[:-1]))
    tr2 = np.maximum(np.abs(low_1d_arr[1:] - close_1d_arr[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, high_1d, atr_14)
    
    # Volume confirmation: current > 1.5x median of last 30 bars
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Volatility filter: avoid extremes (0.5x to 3.0x of median ATR)
        atr_median = pd.Series(atr_14_aligned).rolling(window=50, min_periods=50).median()
        vol_filter = (atr_14_aligned[i] > 0.5 * atr_median[i]) and (atr_14_aligned[i] < 3.0 * atr_median[i])
        
        # Long: Donchian breakout up + volume spike + volatility filter
        if (close[i] > high_20_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Donchian breakout down + volume spike + volatility filter
        elif (close[i] < low_20_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price re-enters Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < high_20_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > low_20_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_DailyDonchian20_Vol1.5x_ATR14Filter"
timeframe = "4h"
leverage = 1.0