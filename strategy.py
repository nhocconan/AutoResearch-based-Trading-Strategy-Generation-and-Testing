#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 10-period high and low for Donchian channels
    high_10 = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Align Donchian levels to 12h timeframe
    upper_12h = align_htf_to_ltf(prices, df_1d, high_10)
    lower_12h = align_htf_to_ltf(prices, df_1d, low_10)
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need Donchian, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_12h[i]) or 
            np.isnan(lower_12h[i]) or 
            np.isnan(ema50_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = close[i] > ema50_12h[i]
        price_below_ema = close[i] < ema50_12h[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND above 1w EMA50 with volume
            if (close[i] > upper_12h[i] and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND below 1w EMA50 with volume
            elif (close[i] < lower_12h[i] and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian lower
            if close[i] < lower_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian upper
            if close[i] > upper_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian10_EMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0