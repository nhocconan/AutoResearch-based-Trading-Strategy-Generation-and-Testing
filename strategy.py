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
    
    # 1h ATR for volatility filter (HTF for 12h strategy)
    df_1h = get_htf_data(prices, '1h')
    tr_1h = np.maximum(df_1h['high'].values - df_1h['low'].values,
                       np.maximum(np.abs(df_1h['high'].values - np.concatenate([[df_1h['close'][0]], df_1h['close'][:-1]])),
                                  np.abs(df_1h['low'].values - np.concatenate([[df_1h['close'][0]], df_1h['close'][:-1]]))))
    atr_1h = pd.Series(tr_1h).rolling(window=14, min_periods=14).mean().values
    atr_1h_aligned = align_htf_to_ltf(prices, df_1h, atr_1h)
    
    # 12h Donchian channels (20-period)
    high_12h = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_12h = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: current > 2.0x median of last 24 bars
    vol_median = pd.Series(volume).rolling(window=24, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    # ATR-based volatility filter: require ATR > 0.3 * median ATR
    atr_median = pd.Series(atr_1h_aligned).rolling(window=50, min_periods=1).median()
    vol_filter = atr_1h_aligned > 0.3 * atr_median
    
    signals = np.zeros(n)
    
    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(vol_filter[i])):
            continue
        
        # Long: close breaks above 12h upper Donchian + volume + volatility filter
        if close[i] > high_12h[i] and volume[i] > vol_threshold[i] and vol_filter[i]:
            signals[i] = 0.25
        
        # Short: close breaks below 12h lower Donchian + volume + volatility filter
        elif close[i] < low_12h[i] and volume[i] > vol_threshold[i] and vol_filter[i]:
            signals[i] = -0.25
        
        # Exit: close crosses back inside Donchian channels (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < high_12h[i]) or
               (signals[i-1] == -0.25 and close[i] > low_12h[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_Breakout_Volume_VolFilter"
timeframe = "12h"
leverage = 1.0