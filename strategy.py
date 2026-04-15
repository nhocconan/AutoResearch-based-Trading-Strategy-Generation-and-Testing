#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action above/below 12h KAMA with 12h volume confirmation
# KAMA adapts to market noise - effective in both trending and ranging markets
# 12h volume > 1.5x median confirms institutional participation
# Conservative sizing (0.25) to manage drawdowns in volatile markets
# Designed for trend following with noise reduction

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h KAMA (adaptive moving average)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Efficiency Ratio calculation
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.abs(np.diff(close_12h))
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(close_12h, np.nan)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # 12h volume confirmation
    vol_12h = df_12h['volume'].values
    vol_median = pd.Series(vol_12h).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
    vol_threshold_aligned = align_htf_to_ltf(prices, df_12h, vol_threshold.values)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(vol_12h_aligned[i]) or 
            np.isnan(vol_threshold_aligned[i])):
            continue
        
        # Long: price above KAMA with volume confirmation
        if (close[i] > kama_aligned[i] and 
            vol_12h_aligned[i] > vol_threshold_aligned[i]):
            signals[i] = 0.25
        
        # Short: price below KAMA with volume confirmation
        elif (close[i] < kama_aligned[i] and 
              vol_12h_aligned[i] > vol_threshold_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price crosses KAMA in opposite direction
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < kama_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > kama_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_KAMA_VolumeConfirmation_12h"
timeframe = "4h"
leverage = 1.0