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
    
    # 4h Donchian(20) channels
    high_4h = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_4h = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_12h_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Volume confirmation: current > 1.5x median of last 30 bars
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or
            np.isnan(ema_12h_50_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Donchian breakout up + above 12h EMA50 + volume spike
        if (close[i] > high_4h[i] and 
            close[i] > ema_12h_50_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Donchian breakout down + below 12h EMA50 + volume spike
        elif (close[i] < low_4h[i] and 
              close[i] < ema_12h_50_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price re-enters Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < high_4h[i]) or
               (signals[i-1] == -0.25 and close[i] > low_4h[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian20_12hEMA50_Volume1.5x"
timeframe = "4h"
leverage = 1.0