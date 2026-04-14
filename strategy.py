#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index + 1-day Donchian breakout
# Long when price breaks above 1-day Donchian upper band AND market is trending (Choppiness < 38.2)
# Short when price breaks below 1-day Donchian lower band AND market is trending (Choppiness < 38.2)
# Exit when price returns to the middle of the Donchian channel
# Uses 1-day Choppiness Index as regime filter to avoid false breakouts in ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing trends

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_high + donchian_low) / 2
    
    # Calculate 1-day True Range for Choppiness Index
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close, 1))
    tr3 = np.abs(low_1d - np.roll(close, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar: no previous close
    tr2[0] = 0  # First bar: no previous close
    tr3[0] = 0  # First bar: no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate 1-day Choppiness Index (14-period)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Align indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (20 for Donchian, 14 for Chop)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: price breaks above Donchian high AND market is trending (Choppiness < 38.2)
            if (price > donchian_high_aligned[i] and chop_aligned[i] < 38.2):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below Donchian low AND market is trending (Choppiness < 38.2)
            elif (price < donchian_low_aligned[i] and chop_aligned[i] < 38.2):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian middle
            if price < donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian middle
            if price > donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Choppiness_DonchianBreakout"
timeframe = "12h"
leverage = 1.0