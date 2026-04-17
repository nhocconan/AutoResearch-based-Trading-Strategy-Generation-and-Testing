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
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian upper and lower bands
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    upper_20_4h = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_4h = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need Donchian, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_4h[i]) or 
            np.isnan(lower_20_4h[i]) or 
            np.isnan(ema50_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = close[i] > ema50_4h[i]
        price_below_ema = close[i] < ema50_4h[i]
        
        if position == 0:
            # Long: Close above Donchian upper band AND price above 1w EMA50 with volume
            if (close[i] > upper_20_4h[i] and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Close below Donchian lower band AND price below 1w EMA50 with volume
            elif (close[i] < lower_20_4h[i] and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close below Donchian lower band OR price crosses below 1w EMA50
            if (close[i] < lower_20_4h[i]) or (close[i] < ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close above Donchian upper band OR price crosses above 1w EMA50
            if (close[i] > upper_20_4h[i]) or (close[i] > ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1wEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0