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
    
    # 1-day ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]])),
                                  np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]]))))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1-week Donchian channels (20-week high/low)
    df_1w = get_htf_data(prices, '1w')
    highest_20w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    lowest_20w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    highest_20w_aligned = align_htf_to_ltf(prices, df_1w, highest_20w)
    lowest_20w_aligned = align_htf_to_ltf(prices, df_1w, lowest_20w)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    # ATR-based volatility filter: require ATR > 0.3 * median ATR
    atr_median = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=1).median()
    vol_filter = atr_1d_aligned > 0.3 * atr_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20w_aligned[i]) or np.isnan(lowest_20w_aligned[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(vol_filter[i])):
            continue
        
        # Long: close breaks above 20-week high + volume + volatility filter
        if close[i] > highest_20w_aligned[i] and volume[i] > vol_threshold[i] and vol_filter[i]:
            signals[i] = 0.25
        
        # Short: close breaks below 20-week low + volume + volatility filter
        elif close[i] < lowest_20w_aligned[i] and volume[i] > vol_threshold[i] and vol_filter[i]:
            signals[i] = -0.25
        
        # Exit: close crosses back inside the 20-week range (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < highest_20w_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > lowest_20w_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_VolFilter"
timeframe = "1d"
leverage = 1.0