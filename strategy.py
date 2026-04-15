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
    
    # --- 12h Donchian breakout (20-period) ---
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max()
    donch_low = low_series.rolling(window=20, min_periods=20).min()
    
    # --- 1d EMA50 trend filter ---
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- Volume confirmation ---
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above Donchian high + above 1d EMA50 + volume spike
        if (close[i] > donch_high[i] and 
            close[i] > ema_50_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low + below 1d EMA50 + volume spike
        elif (close[i] < donch_low[i] and 
              close[i] < ema_50_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back inside Donchian channel (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < donch_high[i-1]) or
               (signals[i-1] == -0.25 and close[i] > donch_low[i-1]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0