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
    
    # Get 1d data for Donchian channels and volume filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on daily data
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume on daily data
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align Donchian levels and volume MA to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    volume_ma20_12h = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need Donchian, EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_12h[i]) or 
            np.isnan(donchian_low_12h[i]) or 
            np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(volume_ma20_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-day average volume
        volume_filter = volume[i] > (1.5 * volume_ma20_12h[i])
        
        if position == 0:
            # Long: Price breaks above 20-day Donchian high with volume and above 12h EMA34
            if (close[i] > donchian_high_12h[i] and volume_filter and close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day Donchian low with volume and below 12h EMA34
            elif (close[i] < donchian_low_12h[i] and volume_filter and close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below 20-day Donchian low OR crosses below 12h EMA34
            if (close[i] < donchian_low_12h[i]) or (close[i] < ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above 20-day Donchian high OR crosses above 12h EMA34
            if (close[i] > donchian_high_12h[i]) or (close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_EMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0