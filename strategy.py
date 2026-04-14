#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter with Donchian breakout
# Long when price breaks above Donchian(20) high AND market is trending (CHOP < 38.2)
# Short when price breaks below Donchian(20) low AND market is trending (CHOP < 38.2)
# Exit when price crosses back inside Donchian channel
# Choppiness Index identifies trending vs ranging markets to avoid false breakouts in chop
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 4h (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index (14-period) - values 0-100, <38.2 = trending, >61.8 = ranging
    atr = pd.Series(np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ])).rolling(window=14, min_periods=14).mean().values
    # Prepend first value to align length
    atr = np.concatenate([[np.nan], atr])
    
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range14 = highest_high14 - lowest_low14
    chop = np.where(range14 != 0, 100 * np.log10(sum_atr14 / range14) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Only trade in trending markets (CHOP < 38.2)
            if chop[i] < 38.2:
                # Long setup: breakout above Donchian high
                if price > high_20[i]:
                    position = 1
                    signals[i] = position_size
                # Short setup: breakdown below Donchian low
                elif price < low_20[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No trades in ranging markets
        elif position == 1:
            # Exit long: price falls back below Donchian low (opposite band)
            if price < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above Donchian high (opposite band)
            if price > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Chop_Donchian_Breakout"
timeframe = "4h"
leverage = 1.0