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
    
    # 1d ATR(14) for volatility filter
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    
    # Calculate True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR calculation with proper min_periods
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Weekly median volume for threshold
    weekly = get_htf_data(prices, '1w')
    vol_w = weekly['volume'].values
    vol_median_w = pd.Series(vol_w).rolling(window=4, min_periods=4).median()
    vol_median_w_aligned = align_htf_to_ltf(prices, weekly, vol_median_w)
    
    # Volatility filter: require ATR > 0.7% of price
    vol_filter = atr_14d_aligned > (0.007 * close)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14d_aligned[i]) or np.isnan(vol_median_w_aligned[i]) or 
            np.isnan(vol_filter[i])):
            continue
        
        # Only trade when volatility is sufficient
        if not vol_filter[i]:
            signals[i] = 0.0
            continue
            
        # Volume threshold: 2.5x weekly median volume (more restrictive)
        vol_threshold = 2.5 * vol_median_w_aligned[i]
        
        # Long: Close above prior close + volume spike
        if (close[i] > close[i-1] and 
            volume[i] > vol_threshold):
            signals[i] = 0.25
        
        # Short: Close below prior close + volume spike
        elif (close[i] < close[i-1] and 
              volume[i] > vol_threshold):
            signals[i] = -0.25
        
        # Exit: reverse signal on opposite direction
        elif (close[i] < close[i-1] and signals[i-1] > 0) or \
             (close[i] > close[i-1] and signals[i-1] < 0):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_1d_ATR_Volume_Momentum"
timeframe = "6h"
leverage = 1.0