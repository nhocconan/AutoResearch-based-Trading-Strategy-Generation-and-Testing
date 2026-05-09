#!/usr/bin/env python3
# 1D_TripleConfluence_HTF1w
# Strategy: 1D price action with 1W trend filter using EMA, volume confirmation, and mean reversion
# Long when: price > 1W EMA10 (uptrend) AND price < BB(20,2) lower band AND volume > 1.5x avg volume
# Short when: price < 1W EMA10 (downtrend) AND price > BB(20,2) upper band AND volume > 1.5x avg volume
# Exit when price crosses back through the Bollinger Band middle (20-period SMA)
# Uses mean reversion within the dominant weekly trend to capture reversals in both bull and bear markets
# Designed for low trade frequency (<25/year) with high conviction entries

name = "1D_TripleConfluence_HTF1w"
timeframe = "1d"
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
    
    # Calculate 1W EMA(10) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Calculate Bollinger Bands (20, 2) on 1D
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_middle = sma_20
    
    # Calculate volume filter: 1.5x 20-period average volume
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_avg_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_10_1w_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: uptrend (price > 1W EMA10) + oversold (price < BB lower) + volume spike
            if (close[i] > ema_10_1w_aligned[i] and 
                close[i] < bb_lower[i] and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: downtrend (price < 1W EMA10) + overbought (price > BB upper) + volume spike
            elif (close[i] < ema_10_1w_aligned[i] and 
                  close[i] > bb_upper[i] and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back above BB middle (mean reversion complete)
            if close[i] > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back below BB middle (mean reversion complete)
            if close[i] < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals