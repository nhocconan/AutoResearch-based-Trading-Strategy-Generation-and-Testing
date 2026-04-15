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
    
    # Load 1d data once
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    
    # Calculate True Range for 1d ATR
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR with proper min_periods
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # 12h Donchian channels (20-period) - using rolling window
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume threshold: 2.0x median of last 12 bars
    vol_median = pd.Series(volume).rolling(window=12, min_periods=12).median().values
    vol_threshold = 2.0 * vol_median
    
    # Volatility filter: require ATR > 0.5% of price
    vol_filter = atr_14d_aligned > (0.005 * close)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_threshold[i]) or 
            np.isnan(vol_filter[i])):
            continue
        
        # Only trade when volatility is sufficient (avoid chop)
        if not vol_filter[i]:
            signals[i] = 0.0
            continue
            
        # Long: Break above Donchian high + volume spike
        if (close[i] > donchian_high[i-1] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Break below Donchian low + volume spike
        elif (close[i] < donchian_low[i-1] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: reverse signal on opposite breakout
        elif (close[i] < donchian_low[i-1] and signals[i-1] > 0) or \
             (close[i] > donchian_high[i-1] and signals[i-1] < 0):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_Breakout_Volume_Filter"
timeframe = "12h"
leverage = 1.0