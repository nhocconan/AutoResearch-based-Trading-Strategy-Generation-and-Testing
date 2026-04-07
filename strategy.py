#!/usr/bin/env python3
"""
12h_turtle_soup_with_volume_confirmation
Hypothesis: Turtle Soup pattern (false breakout reversal) on 12h timeframe. Enter long when price breaks below 20-period low then reverses above it with volume confirmation; enter short when price breaks above 20-period high then reverses below it. Uses 1d trend filter (price above/below 200-period EMA) to align with higher timeframe trend. Designed for 15-30 trades/year to minimize fee drag while capturing reversal moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_turtle_soup_with_volume_confirmation"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels for breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(high[i]) or np.isnan(low[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: price closes below 20-period low (stop loss)
            if close[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 20-period high (stop loss)
            if close[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks below 20-period low then reverses above it (bullish trap)
                if (low[i] < low_20[i] and close[i] > low_20[i] and 
                    close[i] > ema_200_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks above 20-period high then reverses below it (bearish trap)
                elif (high[i] > high_20[i] and close[i] < high_20[i] and 
                      close[i] < ema_200_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals