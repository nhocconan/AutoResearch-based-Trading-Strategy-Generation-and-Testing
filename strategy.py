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
    
    # 12h EMA50 for trend filter (primary trend)
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h.values)
    
    # 4h Donchian channel (20-period breakout)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above Donchian upper + above 12h EMA50 + volume
        if (close[i] > high_20[i] and close[i] > ema_12h_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian lower + below 12h EMA50 + volume
        elif (close[i] < low_20[i] and close[i] < ema_12h_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back inside Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < high_20[i]) or
               (signals[i-1] == -0.25 and close[i] > low_20[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_12h_EMA50_Donchian_Volume"
timeframe = "4h"
leverage = 1.0