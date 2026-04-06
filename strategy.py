#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour price action with 1-day support/resistance levels
# Uses 1-day high/low as dynamic support/resistance levels
# Long when price breaks above 1-day high with volume, short when breaks below 1-day low with volume
# Works in both bull and bear markets by trading breakouts of daily levels
# Target: 50-150 total trades over 4 years with 6h timeframe

name = "6h_daily_breakout_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-day data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Align 1-day high/low to 6h timeframe (shifted by 1 day for no look-ahead)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Volume confirmation: 6h volume > 1.8x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.8x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below 1-day low or stoploss (2x ATR approximation)
            daily_range = high_1d_aligned[i] - low_1d_aligned[i]
            if daily_range > 0:
                stop_loss_level = entry_price - 1.5 * daily_range
            else:
                stop_loss_level = entry_price - 1.5 * 0.01  # fallback
            
            if (close[i] < low_1d_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 1-day high or stoploss
            daily_range = high_1d_aligned[i] - low_1d_aligned[i]
            if daily_range > 0:
                stop_loss_level = entry_price + 1.5 * daily_range
            else:
                stop_loss_level = entry_price + 1.5 * 0.01  # fallback
            
            if (close[i] > high_1d_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with volume
            if volume_filter:
                # Long: break above 1-day high
                if close[i] > high_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: break below 1-day low
                elif close[i] < low_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals