#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-period Donchian high AND weekly EMA50 is rising AND volume > 1.5x 20-period average
# Short when price breaks below 20-period Donchian low AND weekly EMA50 is falling AND volume > 1.5x 20-period average
# Uses weekly EMA50 for trend filter to avoid whipsaws in sideways markets
# Volume confirmation ensures breakout has conviction
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-period EMA on weekly close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate EMA50 slope for trend direction (rising/falling)
    ema50_slope = np.zeros_like(ema50_1w_aligned)
    ema50_slope[1:] = ema50_1w_aligned[1:] - ema50_1w_aligned[:-1]
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema50_slope[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above Donchian high AND weekly EMA50 rising AND volume confirmation
        if (close[i] > high_roll[i] and ema50_slope[i] > 0 and volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: price breaks below Donchian low AND weekly EMA50 falling AND volume confirmation
        elif (close[i] < low_roll[i] and ema50_slope[i] < 0 and volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0