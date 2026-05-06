#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Chandelier Exit for trend-following with volume confirmation
# - Uses 1d ATR-based Chandelier Exit (22, 3.0) for trend direction and exit signals
# - Uses 12h volume spike (1.5x 20-period MA) for entry confirmation
# - Enters long when price closes above long Chandelier Exit with volume confirmation
# - Enters short when price closes below short Chandelier Exit with volume confirmation
# - Exits when price reverses and touches the opposite Chandelier Exit
# - Designed to capture major trends while whipsaw-resistant in sideways markets
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "12h_1dChandelier_22_3.0_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Chandelier Exit calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(22) using Wilder's smoothing
    atr = np.zeros_like(tr)
    atr[21] = np.mean(tr[0:22])  # Simple average for first value
    for i in range(22, len(tr)):
        atr[i] = (atr[i-1] * 21 + tr[i]) / 22
    
    # Chandelier Exit calculation (22-period, 3.0 multiplier)
    # Long Exit: highest high - 3*ATR
    # Short Exit: lowest low + 3*ATR
    highest_high = np.zeros_like(high_1d)
    lowest_low = np.zeros_like(low_1d)
    
    highest_high[21] = np.max(high_1d[0:22])
    lowest_low[21] = np.min(low_1d[0:22])
    
    for i in range(22, len(high_1d)):
        highest_high[i] = max(highest_high[i-1], high_1d[i])
        lowest_low[i] = min(lowest_low[i-1], low_1d[i])
    
    long_exit = highest_high - 3.0 * atr
    short_exit = lowest_low + 3.0 * atr
    
    # Align Chandelier Exits to 12h timeframe
    long_exit_12h = align_htf_to_ltf(prices, df_1d, long_exit)
    short_exit_12h = align_htf_to_ltf(prices, df_1d, short_exit)
    
    # Volume filter (12h timeframe)
    vol_ma_20 = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(long_exit_12h[i]) or np.isnan(short_exit_12h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close above long Chandelier Exit with volume confirmation
            if close[i] > long_exit_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: close below short Chandelier Exit with volume confirmation
            elif close[i] < short_exit_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches or crosses short Chandelier Exit
            if close[i] <= short_exit_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches or crosses long Chandelier Exit
            if close[i] >= long_exit_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals