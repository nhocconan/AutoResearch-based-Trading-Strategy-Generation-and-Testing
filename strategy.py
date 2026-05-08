#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Choppiness Index regime filter with 4-hour Donchian breakout
# Long when CHOP > 61.8 (range) + price breaks above Donchian high (20) + volume confirmation
# Short when CHOP > 61.8 (range) + price breaks below Donchian low (20) + volume confirmation
# Choppiness Index identifies ranging markets ideal for mean-reversion breakout strategies
# Donchian breakout provides clear entry/exit levels with built-in trend following
# Volume confirmation ensures institutional participation in breakouts
# Works in both bull and bear markets by adapting to ranging conditions
# Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "6h_Chop_Donchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data once for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate Choppiness Index (14-period) on 6h data
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10((atr * 14) / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)
    
    # Volume spike: current volume > 1.5 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        chop_val = chop[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: ranging market + breakout above Donchian high + volume
            if chop_val > 61.8 and close[i] > dh and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: ranging market + breakdown below Donchian low + volume
            elif chop_val > 61.8 and close[i] < dl and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: breakdown below Donchian low OR chop drops below 38.2 (trending)
            if close[i] < dl or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: breakout above Donchian high OR chop drops below 38.2 (trending)
            if close[i] > dh or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals