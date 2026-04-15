#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d ATR filter
# Uses 12h timeframe to reduce trade frequency and fee drag, targeting 12-37 trades/year.
# Donchian breakout captures trends; volume ensures institutional participation.
# 1d ATR filter avoids low volatility (chop) and exhaustion volatility.
# Designed for trend following in both bull and bear markets with conservative sizing (0.25).

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
    
    # 12h Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 1d ATR(10) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_10_aligned = align_htf_to_ltf(prices, df_1d, atr_10)
    
    # Volume confirmation: current > 1.8x median of last 50 bars
    vol_median = pd.Series(volume).rolling(window=50, min_periods=1).median()
    vol_threshold = 1.8 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(atr_10_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Volatility filter: avoid extremes
        atr_median = pd.Series(atr_10_aligned).rolling(window=100, min_periods=100).median()
        vol_filter = (atr_10_aligned[i] > 0.3 * atr_median[i]) and (atr_10_aligned[i] < 3.0 * atr_median[i])
        
        # Long: Donchian breakout up + volume spike + volatility filter
        if (close[i] > high_20[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Donchian breakout down + volume spike + volatility filter
        elif (close[i] < low_20[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price re-enters Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < high_20[i]) or
               (signals[i-1] == -0.25 and close[i] > low_20[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_DonchianBreakout20_Volume1.8x_ATRFilter"
timeframe = "12h"
leverage = 1.0