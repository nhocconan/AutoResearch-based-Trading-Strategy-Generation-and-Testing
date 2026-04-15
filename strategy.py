#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1-day high/low/close with volume confirmation
# Camarilla levels provide precise support/resistance based on previous day's range.
# Long when price crosses above L3 with volume confirmation; short when below H3.
# Volume > 2.0x 20-bar median ensures institutional participation.
# Works in both bull (breakouts above resistance) and bear (breakdowns below support).
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1-day bar
    # H3 = close + 1.1 * (high - low) / 6
    # L3 = close - 1.1 * (high - low) / 6
    rang = high_1d - low_1d
    H3 = close_1d + 1.1 * rang / 6
    L3 = close_1d - 1.1 * rang / 6
    
    # Align Camarilla levels to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current > 2.0x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: price crosses above L3 with volume confirmation
        if (close[i] > L3_aligned[i] and 
            (i == 0 or close[i-1] <= L3_aligned[i]) and  # crossed above
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price crosses below H3 with volume confirmation
        elif (close[i] < H3_aligned[i] and 
              (i == 0 or close[i-1] >= H3_aligned[i]) and  # crossed below
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price returns to mid-point between H3 and L3
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (H3_aligned[i] + L3_aligned[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (H3_aligned[i] + L3_aligned[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_CamarillaPivot_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0