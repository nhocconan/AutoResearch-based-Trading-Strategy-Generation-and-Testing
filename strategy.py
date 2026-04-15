#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly pivot-based trend filter + volume spike
# Uses weekly high/low to establish trend context and avoid counter-trend trades
# Volume spike confirms institutional participation in breakouts
# Designed for low trade frequency (12-37/year) to minimize fee drag
# Works in bull/bear by aligning with higher timeframe momentum

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend context (called ONCE before loop)
    weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly high and low for trend determination
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    
    # Align weekly high/low to 6s timeframe (waits for weekly bar close)
    weekly_high_aligned = align_htf_to_ltf(prices, weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, weekly, weekly_low)
    
    # Weekly average volume for spike detection
    weekly_vol = weekly['volume'].values
    vol_ma_4w = pd.Series(weekly_vol).rolling(window=4, min_periods=4).mean().values
    vol_ma_4w_aligned = align_htf_to_ltf(prices, weekly, vol_ma_4w)
    
    # Volume spike threshold
    vol_threshold = 1.5 * vol_ma_4w_aligned
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if weekly data not yet available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(vol_ma_4w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long: Price breaks above weekly high with volume spike
        if (close[i] > weekly_high_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Price breaks below weekly low with volume spike
        elif (close[i] < weekly_low_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price returns to weekly midpoint (mean reversion within weekly range)
        elif (weekly_high_aligned[i] > weekly_low_aligned[i]):  # avoid division by zero
            weekly_mid = (weekly_high_aligned[i] + weekly_low_aligned[i]) / 2
            if ((signals[i-1] > 0 and close[i] < weekly_mid) or
                (signals[i-1] < 0 and close[i] > weekly_mid)):
                signals[i] = 0.0
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_WeeklyHighLow_Breakout_Volume"
timeframe = "6h"
leverage = 1.0