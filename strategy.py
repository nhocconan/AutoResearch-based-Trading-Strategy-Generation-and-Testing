# 6H_Weekly_Pivot_Trend_Confirmation_v1
# Hypothesis: Use 1-week pivot point direction (bullish/bearish bias) as trend filter and 6h price action for entry.
# Long when price is above weekly pivot AND breaks above 6h high of prior 3 periods with volume confirmation.
# Short when price is below weekly pivot AND breaks below 6h low of prior 3 periods with volume confirmation.
# Weekly pivot provides structural bias to avoid counter-trend trades in both bull and bear markets.
# Volume confirmation ensures momentum behind breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee churn.

name = "6H_Weekly_Pivot_Trend_Confirmation_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1w data for weekly pivot (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    # Weekly pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    # Align to 6t timeframe (wait for weekly close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # 6h price action: highest high and lowest low of prior 3 periods
    # Use rolling window of 3 for recent swing points
    high_roll = pd.Series(high).rolling(window=3, min_periods=3).max().values
    low_roll = pd.Series(low).rolling(window=3, min_periods=3).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(20, 3)  # Ensure sufficient warmup for volume avg and roll
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 6 bars between trades (~1.5 days on 6h TF) to reduce frequency
            if bars_since_exit < 6:
                continue
                
            # Long: price above weekly pivot AND breaks above 6h high of prior 3 periods
            if (close[i] > pivot_1w_aligned[i] and 
                high[i] > high_roll[i-1] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price below weekly pivot AND breaks below 6h low of prior 3 periods
            elif (close[i] < pivot_1w_aligned[i] and 
                  low[i] < low_roll[i-1] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite side of weekly pivot
            if position == 1 and close[i] < pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals