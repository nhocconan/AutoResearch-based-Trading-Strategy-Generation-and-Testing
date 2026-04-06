#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price action strategy using Bollinger Bands squeeze breakout with volume confirmation
# Long when Bollinger Bands width < 20th percentile AND price breaks above upper band AND volume > 1.5x average
# Short when Bollinger Bands width < 20th percentile AND price breaks below lower band AND volume > 1.5x average
# Exit when price crosses middle band (20-period SMA)
# Bollinger Bands squeeze indicates low volatility, often preceding explosive moves
# Volume confirmation ensures breakout has conviction
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 12h performance

name = "12h_bb_squeeze_breakout_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20-period, 2 std dev)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Bollinger Band Width (normalized by middle band)
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Calculate 20th percentile of BB width for squeeze condition
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20).values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses middle band
        if position == 1:  # long position
            if close[i] < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Bollinger squeeze and volume confirmation
            # Long: BB squeeze AND price breaks above upper band AND volume confirmation
            if (bb_width[i] < bb_width_percentile[i] and 
                close[i] > bb_upper[i] and close[i-1] <= bb_upper[i-1] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze AND price breaks below lower band AND volume confirmation
            elif (bb_width[i] < bb_width_percentile[i] and 
                  close[i] < bb_lower[i] and close[i-1] >= bb_lower[i-1] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>