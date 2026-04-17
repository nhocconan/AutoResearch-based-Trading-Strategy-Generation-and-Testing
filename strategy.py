#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivots: P = (H+L+C)/3, S1 = 2*P - H, R1 = 2*P - L
    pivot = (high_1d + low_1d + close_1d) / 3
    s1 = 2 * pivot - high_1d
    r1 = 2 * pivot - low_1d
    
    # Align daily pivots to 4h timeframe
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: 4h close above/below 50-period EMA
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need EMA50 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(s1_4h[i]) or 
            np.isnan(r1_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        if position == 0:
            # Long: price crosses above R1 with volume and above EMA50
            if close[i] > r1_4h[i] and volume_filter and close[i] > ema50[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below S1 with volume and below EMA50
            elif close[i] < s1_4h[i] and volume_filter and close[i] < ema50[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below pivot or volume drops
            if close[i] < pivot[i] or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above pivot or volume drops
            if close[i] > pivot[i] or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1_S1_VolumeTrend"
timeframe = "4h"
leverage = 1.0