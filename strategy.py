#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute hour filter
    hours = prices.index.hour
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h timeframe for trend
    daily = get_htf_data(prices, '4h')
    close_4h = daily['close'].values
    high_4h = daily['high'].values
    low_4h = daily['low'].values
    
    # 4h EMA(21) for trend
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, daily, ema_21_4h)
    
    # 1h EMA(50) for entry filter
    ema_50_1h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: 1.5x 20-period median
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_50_1h[i]) or 
            np.isnan(vol_threshold[i])):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # Long: 4h uptrend + price > 1h EMA50 + volume spike
        if (close[i] > ema_50_1h[i] and 
            ema_21_4h_aligned[i] > ema_21_4h_aligned[i-1] and  # 4h EMA rising
            volume[i] > vol_threshold[i]):
            signals[i] = 0.20
        
        # Short: 4h downtrend + price < 1h EMA50 + volume spike
        elif (close[i] < ema_50_1h[i] and 
              ema_21_4h_aligned[i] < ema_21_4h_aligned[i-1] and  # 4h EMA falling
              volume[i] > vol_threshold[i]):
            signals[i] = -0.20
        
        # Exit: price crosses back below/above 1h EMA50
        elif (i > 0 and 
              ((signals[i-1] == 0.20 and close[i] < ema_50_1h[i]) or
               (signals[i-1] == -0.20 and close[i] > ema_50_1h[i]))):
            signals[i] = 0.0
        
        # Otherwise hold
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1h_4h_EMA21_Vol1.5x_EMA50_Filter"
timeframe = "1h"
leverage = 1.0