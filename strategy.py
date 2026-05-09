#!/usr/bin/env python3
# Hypothesis: 6h timeframe with 12h Bollinger Band squeeze breakout and volume confirmation.
# Uses Bollinger Band width (BBW) percentile to detect low volatility squeeze conditions.
# Breakouts occur when price closes outside Bollinger Bands after a squeeze (BBW < 20th percentile).
# Volume confirmation ensures breakout validity. Works in both bull and bear markets by capturing
# volatility expansion phases regardless of direction. Target: 50-150 total trades over 4 years.

name = "6h_Bollinger_Squeeze_Breakout_12hBBW_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Bollinger Bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Bollinger Bands (20, 2)
    close_12h = df_12h['close'].values
    sma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Bollinger Band Width
    bbw = (upper_bb - lower_bb) / sma_20
    
    # Bollinger Band Width percentile (20-period lookback)
    bbw_percentile = pd.Series(bbw).rolling(window=20, min_periods=20).rank(pct=True).values
    
    # Squeeze condition: BBW below 20th percentile
    squeeze = bbw_percentile < 0.2
    
    # Breakout conditions: price closes outside Bollinger Bands
    breakout_up = close_12h > upper_bb
    breakout_down = close_12h < lower_bb
    
    # Align 12h indicators to 6h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_12h, squeeze)
    breakout_up_aligned = align_htf_to_ltf(prices, df_12h, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_12h, breakout_down)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(squeeze_aligned[i]) or np.isnan(breakout_up_aligned[i]) or 
            np.isnan(breakout_down_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: squeeze breakout up + volume filter
            if squeeze_aligned[i] and breakout_up_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakout down + volume filter
            elif squeeze_aligned[i] and breakout_down_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle Bollinger Band (SMA20) or squeeze ends
            if close_12h[i] <= sma_20[i] or not squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle Bollinger Band (SMA20) or squeeze ends
            if close_12h[i] >= sma_20[i] or not squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals