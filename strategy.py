#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1w_volume_v1"
timezone = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel (breakout signal)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper and lower bands
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_upper_12h = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_12h = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions: price crosses back through Donchian levels or volume fade
        exit_long = (close[i] < donchian_lower_12h[i]) or (not vol_confirm)
        exit_short = (close[i] > donchian_upper_12h[i]) or (not vol_confirm)
        
        if position == 1:  # Long position
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: Price breaks above Donchian upper + volume confirmation
            if close[i] > donchian_upper_12h[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: Price breaks below Donchian lower + volume confirmation
            elif close[i] < donchian_lower_12h[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals