# state your hypothesis in a comment
# Hypothesis: 4h Donchian(10) breakout with volume confirmation and 1d ATR filter
# Shorter Donchian period (10) for more frequent but still controlled signals.
# Volume > 1.8x median ensures significant participation.
# 1d ATR(10) filter avoids low volatility (chop) and exhaustion volatility.
# Designed for trend following in both bull and bear markets with conservative sizing.
# Target: 20-50 trades/year per symbol to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(10) channels
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max()
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min()
    
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
    
    for i in range(10, n):
        # Skip if any required data is NaN
        if (np.isnan(high_10[i]) or np.isnan(low_10[i]) or
            np.isnan(atr_10_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Volatility filter: avoid extremes
        atr_median = pd.Series(atr_10_aligned).rolling(window=100, min_periods=100).median()
        vol_filter = (atr_10_aligned[i] > 0.3 * atr_median[i]) and (atr_10_aligned[i] < 3.0 * atr_median[i])
        
        # Long: Donchian breakout up + volume spike + volatility filter
        if (close[i] > high_10[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Donchian breakout down + volume spike + volatility filter
        elif (close[i] < low_10[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price re-enters Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < high_10[i]) or
               (signals[i-1] == -0.25 and close[i] > low_10[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_DonchianBreakout10_Volume1.8x_ATRFilter"
timeframe = "4h"
leverage = 1.0