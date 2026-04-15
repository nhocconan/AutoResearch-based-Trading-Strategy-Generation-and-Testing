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
    
    # Daily data for 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    
    # Daily high, low, close for Camarilla pivot calculation
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    camarilla_h4 = daily_close + 1.5 * (daily_high - daily_low)
    camarilla_l4 = daily_close - 1.5 * (daily_high - daily_low)
    
    # Align to 12h timeframe - previous day's levels available after daily close
    camarilla_h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h4_12h[i]) or np.isnan(camarilla_l4_12h[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: price touches or goes below L4 + volume confirmation
        if low[i] <= camarilla_l4_12h[i] and volume[i] > vol_threshold[i]:
            signals[i] = 0.25
        
        # Short: price touches or goes above H4 + volume confirmation
        elif high[i] >= camarilla_h4_12h[i] and volume[i] > vol_threshold[i]:
            signals[i] = -0.25
        
        # Exit: price moves back toward midpoint (mean reversion)
        elif i > 0 and signals[i-1] != 0:
            midpoint = (camarilla_h4_12h[i] + camarilla_l4_12h[i]) / 2
            if (signals[i-1] == 0.25 and close[i] >= midpoint) or \
               (signals[i-1] == -0.25 and close[i] <= midpoint):
                signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Camarilla_Pivot_Touch_Volume"
timeframe = "12h"
leverage = 1.0