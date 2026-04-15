# 4h_PostClose_Reversal_With_Volume
# Reversal after close of Bollinger band breach with volume confirmation
# Works in both bull and bear by fading extremes during volatility spikes
# Target: 30-80 trades/year, avoids overtrading

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
    
    # Bollinger Bands (20, 2) on close
    close_series = pd.Series(close)
    sma = close_series.rolling(window=20, min_periods=20).mean()
    std = close_series.rolling(window=20, min_periods=20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    
    # Volume filter: current volume > 1.8x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.8 * vol_median
    
    # Bollinger Band width for volatility regime
    bb_width = (upper - lower) / sma
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean()
    vol_expansion = bb_width > bb_width_ma  # Volatility expanding
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(vol_expansion[i])):
            continue
        
        # Long: Price closed ABOVE upper band PREVIOUS bar + volume + vol expansion
        if (i > 0 and close[i-1] > upper[i-1] and 
            volume[i] > vol_threshold[i] and vol_expansion[i]):
            signals[i] = 0.25
        
        # Short: Price closed BELOW lower band PREVIOUS bar + volume + vol expansion
        elif (i > 0 and close[i-1] < lower[i-1] and 
              volume[i] > vol_threshold[i] and vol_expansion[i]):
            signals[i] = -0.25
        
        # Exit: Price returns to middle (SMA) or opposite band touched
        elif i > 0:
            if signals[i-1] == 0.25 and close[i] <= sma[i]:
                signals[i] = 0.0
            elif signals[i-1] == -0.25 and close[i] >= sma[i]:
                signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_PostClose_Reversal_With_Volume"
timeframe = "4h"
leverage = 1.0