#!/usr/bin/env python3
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
    
    # Volume confirmation (10-period MA on 4h)
    volume_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # Get 1d data for trend filter and breakout levels
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d rolling high (10 days) and low (10 days)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    high_10_1d = high_1d_series.rolling(window=10, min_periods=10).max().values
    low_10_1d = low_1d_series.rolling(window=10, min_periods=10).min().values
    
    # Align 1d high/low to 4h timeframe
    high_10_1d_aligned = align_htf_to_ltf(prices, df_1d, high_10_1d)
    low_10_1d_aligned = align_htf_to_ltf(prices, df_1d, low_10_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(10, 10, 50)  # volume MA10, 1d high/low lookback, EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma10[i]) or 
            np.isnan(high_10_1d_aligned[i]) or 
            np.isnan(low_10_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 10-period average
        volume_filter = volume[i] > (1.5 * volume_ma10[i])
        
        if position == 0:
            # Long: price > 1d high (10) + volume filter + 1d uptrend
            if close[i] > high_10_1d_aligned[i] and volume_filter and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < 1d low (10) + volume filter + 1d downtrend
            elif close[i] < low_10_1d_aligned[i] and volume_filter and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < 1d low (10) or 1d trend turns down
            if close[i] < low_10_1d_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > 1d high (10) or 1d trend turns up
            if close[i] > high_10_1d_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_HighLowBreakout_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0