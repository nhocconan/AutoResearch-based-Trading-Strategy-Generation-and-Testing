#!/usr/bin/env python3
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
    
    # 1w high/low for Donchian channel (weekly)
    weekly = get_htf_data(prices, '1w')
    high_w = weekly['high'].values
    low_w = weekly['low'].values
    donchian_high = pd.Series(high_w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_w).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, weekly, donchian_low)
    
    # 1d ATR(14) for volatility filter
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    tr1 = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1]))
    tr2 = np.maximum(np.abs(low_d[1:] - close_d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Volume threshold: 2.0x median of last 50 bars (more selective)
    vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median()
    vol_threshold = 2.0 * vol_median
    
    # ATR median for volatility regime filter
    atr_median = pd.Series(atr_14d_aligned).rolling(window=100, min_periods=100).median()
    
    signals = np.zeros(n)
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_14d_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(atr_median[i])):
            continue
        
        # Volatility filter: avoid extremes (0.7x to 2.5x of median ATR)
        vol_filter = (atr_14d_aligned[i] > 0.7 * atr_median[i]) and (atr_14d_aligned[i] < 2.5 * atr_median[i])
        
        # Long: Price breaks above weekly Donchian high + volume spike + volatility filter
        if (close[i] > donchian_high_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Price breaks below weekly Donchian low + volume spike + volatility filter
        elif (close[i] < donchian_low_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price crosses back below/above weekly Donchian levels
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < donchian_low_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > donchian_high_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_WeeklyDonchian20_Vol2.0x_ATR14dFilter_v1"
timeframe = "12h"
leverage = 1.0