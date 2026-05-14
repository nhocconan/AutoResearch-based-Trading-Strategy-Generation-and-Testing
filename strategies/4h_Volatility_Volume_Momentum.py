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
    
    # 1d data for ATR filter (once before loop)
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    
    # True Range calculation
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR with proper min_periods
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Volume filter: 2.0x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 2.0 * vol_median
    
    # ATR filter: require ATR > 0.5% of price (avoid low volatility)
    vol_filter = atr_14d_aligned > (0.005 * close)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14d_aligned[i]) or np.isnan(vol_threshold[i]) or 
            np.isnan(vol_filter[i])):
            continue
        
        # Only trade when volatility is sufficient (avoid chop)
        if not vol_filter[i]:
            signals[i] = 0.0
            continue
            
        # Long: Close above prior close + volume spike
        if (close[i] > close[i-1] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Close below prior close + volume spike
        elif (close[i] < close[i-1] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: reverse signal on opposite direction
        elif (close[i] < close[i-1] and signals[i-1] > 0) or \
             (close[i] > close[i-1] and signals[i-1] < 0):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Volatility_Volume_Momentum"
timeframe = "4h"
leverage = 1.0