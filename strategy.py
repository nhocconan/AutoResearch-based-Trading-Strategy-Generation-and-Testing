#!/usr/bin/env python3
"""
12h_1w_SMA_Crossover_Volume_Confirmation_v1
Hypothesis: Uses 1-week SMA crossover (SMA50/SMA200) for trend direction, 
with 12h price action confirming breakouts above/below recent swing points,
and volume surge for entry. Designed for low trade frequency (12-37/year) 
to work in both bull and bear markets by capturing major trend changes 
with institutional volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend determination
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly SMAs for trend filter
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma200_1w = pd.Series(close_1w).rolling(window=200, min_periods=200).mean().values
    
    # Align weekly SMAs to 12h timeframe
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    sma200_1w_aligned = align_htf_to_ltf(prices, df_1w, sma200_1w)
    
    # Calculate 12h swing points for entry levels
    # Use 24-period lookback for swing high/low (equivalent to 12 days on 12h)
    lookback = 24
    high_max = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    low_min = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2.5 * 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Start after all indicators are ready
    start_idx = max(200, lookback, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(sma50_1w_aligned[i]) or np.isnan(sma200_1w_aligned[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        sma50 = sma50_1w_aligned[i]
        sma200 = sma200_1w_aligned[i]
        swing_high = high_max[i]
        swing_low = low_min[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine trend from weekly SMA crossover
            bullish_trend = sma50 > sma200
            bearish_trend = sma50 < sma200
            
            if bullish_trend and vol_conf:
                # Long: break above recent swing high with volume
                if close_val > swing_high:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif bearish_trend and vol_conf:
                # Short: break below recent swing low with volume
                if close_val < swing_low:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: price breaks below swing low or trend changes
            if close_val < swing_low or sma50 < sma200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price breaks above swing high or trend changes
            if close_val > swing_high or sma50 > sma200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_1w_SMA_Crossover_Volume_Confirmation_v1"
timeframe = "12h"
leverage = 1.0