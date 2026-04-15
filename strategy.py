#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on close
    sma = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_threshold[i]):
            continue
        
        # Long: close breaks above upper band + volume confirmation
        if close[i] > upper[i] and volume[i] > vol_threshold[i]:
            signals[i] = 0.25
        
        # Short: close breaks below lower band + volume confirmation
        elif close[i] < lower[i] and volume[i] > vol_threshold[i]:
            signals[i] = -0.25
        
        # Exit: close crosses back inside bands (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < upper[i]) or
               (signals[i-1] == -0.25 and close[i] > lower[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Bollinger_Breakout_Volume"
timeframe = "4h"
leverage = 1.0