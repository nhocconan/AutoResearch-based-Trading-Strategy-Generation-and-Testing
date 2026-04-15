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
    
    # Daily high/low for range
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    
    # Daily high/low range (previous day)
    range_high = high_d
    range_low = low_d
    range_high_aligned = align_htf_to_ltf(prices, daily, range_high)
    range_low_aligned = align_htf_to_ltf(prices, daily, range_low)
    
    # Daily ATR for volatility filter
    tr1 = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - daily['close'].values[:-1]))
    tr2 = np.maximum(np.abs(low_d[1:] - daily['close'].values[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_d_aligned = align_htf_to_ltf(prices, daily, atr_d)
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_threshold = 1.5 * vol_ma
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(range_high_aligned[i]) or np.isnan(range_low_aligned[i]) or
            np.isnan(atr_d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Volatility filter: avoid low volatility (< 0.5 * ATR)
        vol_filter = atr_d_aligned[i] > 0.5 * np.nanmedian(atr_d_aligned[max(0, i-50):i+1])
        
        # Long: Break above daily high + volume spike + volatility filter
        if (close[i] > range_high_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.20
        
        # Short: Break below daily low + volume spike + volatility filter
        elif (close[i] < range_low_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.20
        
        # Exit: price returns to mid-range of daily range
        elif (i > 0 and 
              ((signals[i-1] == 0.20 and close[i] < (range_high_aligned[i] + range_low_aligned[i]) / 2) or
               (signals[i-1] == -0.20 and close[i] > (range_high_aligned[i] + range_low_aligned[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1h_DailyRangeBreakout_Vol1.5x_VolFilter_v1"
timeframe = "1h"
leverage = 1.0