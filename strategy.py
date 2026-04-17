# The strategy uses 4h timeframe with a 1d high-low range breakout system.
# Long when price breaks above the prior 1d high with volume confirmation.
# Short when price breaks below the prior 1d low with volume confirmation.
# Exit when price returns to the prior 1d close level.
# Position size: 0.25. Designed to capture momentum with controlled frequency.
# Works in both bull and bear markets by trading breakouts in the direction of the prior day's range.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 2:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for prior day's high, low, close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate prior day's high, low, close (shifted by 1 to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get prior day's values (avoid look-ahead)
    prior_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prior_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    prior_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Align to 4h timeframe
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high_1d)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low_1d)
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close_1d)
    
    # Volume filter: current volume > 1.5x 20-period moving average
    volume_ma20 = np.convolve(volume, np.ones(20)/20, mode='full')[:len(volume)]
    volume_ma20 = np.concatenate([np.full(19, np.nan), volume_ma20[19:]])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(prior_high_aligned[i]) or np.isnan(prior_low_aligned[i]) or 
            np.isnan(prior_close_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Entry signals
        if position == 0:
            # Long: Break above prior day's high with volume confirmation
            if close[i] > prior_high_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below prior day's low with volume confirmation
            elif close[i] < prior_low_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price returns to prior day's close
            if close[i] <= prior_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price returns to prior day's close
            if close[i] >= prior_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4d_PriorDayRange_Breakout_Volume"
timeframe = "4h"
leverage = 1.0