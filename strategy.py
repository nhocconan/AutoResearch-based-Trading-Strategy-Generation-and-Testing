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
    
    # Weekly 20-period Donchian channel (breakout structure)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    high_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max()
    low_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min()
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Weekly 14-period ATR for volatility filter
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.maximum(np.abs(low_1w[1:] - close_1w[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14w_aligned = align_htf_to_ltf(prices, df_1w, atr_14w)
    
    # Volume confirmation: current > 2.0x median of last 30 bars
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or
            np.isnan(atr_14w_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Volatility filter: avoid extremes (0.5x to 3.0x of median ATR)
        atr_median = pd.Series(atr_14w_aligned).rolling(window=50, min_periods=50).median()
        vol_filter = (atr_14w_aligned[i] > 0.5 * atr_median[i]) and (atr_14w_aligned[i] < 3.0 * atr_median[i])
        
        # Long: Weekly Donchian breakout up + volume spike + volatility filter
        if (close[i] > high_20w_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Weekly Donchian breakout down + volume spike + volatility filter
        elif (close[i] < low_20w_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price re-enters weekly Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < high_20w_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > low_20w_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyDonchian20_Vol2.0x_ATR14Filter"
timeframe = "1d"
leverage = 1.0