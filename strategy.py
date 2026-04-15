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
    
    # 4h Donchian breakout (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_donchian = high_series.rolling(window=20, min_periods=20).max()
    lower_donchian = low_series.rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    # Daily EMA trend filter (1d EMA 50)
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(ema_1d_aligned[i])):
            continue
        
        # Long: price breaks above upper Donchian + volume confirmation + above daily EMA
        if (close[i] > upper_donchian[i] and volume[i] > vol_threshold[i] and 
            close[i] > ema_1d_aligned[i]):
            signals[i] = 0.25
        
        # Short: price breaks below lower Donchian + volume confirmation + below daily EMA
        elif (close[i] < lower_donchian[i] and volume[i] > vol_threshold[i] and 
              close[i] < ema_1d_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back to the opposite Donchian band
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < lower_donchian[i]) or
               (signals[i-1] == -0.25 and close[i] > upper_donchian[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Breakout_Volume_EMA"
timeframe = "4h"
leverage = 1.0