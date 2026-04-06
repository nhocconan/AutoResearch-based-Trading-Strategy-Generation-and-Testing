#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with 1d volume confirmation
# Long when price breaks above upper BB(20,2) AND volume > 1.5x average AND BB width < 20th percentile (squeeze)
# Short when price breaks below lower BB(20,2) AND volume > 1.5x average AND BB width < 20th percentile
# Exit when price returns to middle BB OR BB width > 50th percentile (squeeze ended)
# Works in both bull/bear markets by capturing breakouts from low volatility periods

name = "12h_bb_squeeze_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 12h
    sma = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    middle = sma
    
    upper = upper.values
    lower = lower.values
    middle = middle.values
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper - lower) / middle
    bb_width = np.where(middle != 0, bb_width, 0)
    
    # Percentile of BB width (20-period lookback) to identify squeeze
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume confirmation from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    daily_volume = df_1d['volume'].values
    
    # Calculate average daily volume
    avg_daily_volume = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * avg_daily_volume.values
    
    # Align daily volume threshold to 12h timeframe
    volume_threshold_aligned = align_htf_to_ltf(prices, df_1d, volume_threshold)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or \
           np.isnan(bb_width_percentile[i]) or np.isnan(volume_threshold_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price returns to middle OR squeeze ends (BB width > 50th percentile)
        if position == 1:  # long position
            if close[i] <= middle[i] or bb_width_percentile[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= middle[i] or bb_width_percentile[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts during squeeze (low volatility)
            # Long: price breaks above upper BB during squeeze + volume confirmation
            if (close[i] > upper[i] and bb_width_percentile[i] < 20 and 
                volume[i] > volume_threshold_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB during squeeze + volume confirmation
            elif (close[i] < lower[i] and bb_width_percentile[i] < 20 and 
                  volume[i] > volume_threshold_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals