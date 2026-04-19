#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter
# Long when price breaks above Donchian upper AND volume > 1.5x 20-period average AND chop > 61.8 (range)
# Short when price breaks below Donchian lower AND volume > 1.5x 20-period average AND chop > 61.8
# Exit when price crosses opposite Donchian band OR chop < 38.2 (trending)
# Uses Donchian for breakout structure, volume for confirmation, chop for regime filter.
# Target: 20-50 trades/year per symbol.
name = "4h_Donchian_Breakout_Volume_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Chop index (14-period)
    atr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values
    atr[0] = high[0] - low[0]  # Fix first value
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((highest_high - lowest_low) / (np.cumsum(atr) - np.roll(np.cumsum(atr), 1))) / np.log10(14)
    chop = np.where(np.isnan(chop), 50, chop)  # Fill NaN with neutral
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure Donchian is ready
    
    for i in range(start_idx, n):
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        vol = volume[i]
        vol_avg = vol_ma[i]
        chop_val = chop[i]
        
        # Skip if any required data is not available
        if np.isnan(upper) or np.isnan(lower) or np.isnan(vol_avg) or np.isnan(chop_val):
            signals[i] = 0.0
            continue
        
        # Volume confirmation (>1.5x average)
        volume_confirmed = vol > 1.5 * vol_avg
        
        # Chop regime (>61.8 = ranging, <38.2 = trending)
        chop_ranging = chop_val > 61.8
        chop_trending = chop_val < 38.2
        
        if position == 0:
            # Long entry: breakout above upper band + volume + ranging
            if price > upper and volume_confirmed and chop_ranging:
                signals[i] = 0.25
                position = 1
            # Short entry: breakout below lower band + volume + ranging
            elif price < lower and volume_confirmed and chop_ranging:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower band OR chop becomes trending
            if price < lower or chop_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper band OR chop becomes trending
            if price > upper or chop_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals