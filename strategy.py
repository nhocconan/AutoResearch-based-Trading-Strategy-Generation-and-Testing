#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian and ATR
    daily = get_htf_data(prices, '1d')
    close_d = daily['close'].values
    high_d = daily['high'].values
    low_d = daily['low'].values
    
    # 1d Donchian channels (20-period)
    donch_high = np.full(len(close_d), np.nan)
    donch_low = np.full(len(close_d), np.nan)
    for i in range(20, len(close_d)):
        donch_high[i] = np.max(high_d[i-20:i])
        donch_low[i] = np.min(low_d[i-20:i])
    donch_high_aligned = align_htf_to_ltf(prices, daily, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, daily, donch_low)
    
    # 1d ATR(14) for volatility filter
    tr = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Volume threshold: 2.0x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(atr_14d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Price breaks above 1d Donchian high + volume spike
        if (close[i] > donch_high_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Price breaks below 1d Donchian low + volume spike
        elif (close[i] < donch_low_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle of Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (donch_high_aligned[i] + donch_low_aligned[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (donch_high_aligned[i] + donch_low_aligned[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_1d_Donchian20_Vol2.0x_Breakout"
timeframe = "4h"
leverage = 1.0