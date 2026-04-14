#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator (Jaw/Teeth/Lips) combined with 12-hour high-low range filter
# Long when price is above alligator lines AND above 12h 20-period high (bullish structure)
# Short when price is below alligator lines AND below 12h 20-period low (bearish structure)
# Exit when price crosses the alligator teeth (middle line)
# Williams Alligator identifies trend phases, 12h high/low filter ensures alignment with higher timeframe structure
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing trends

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h and 12h data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Williams Alligator on 6h: Jaw(13,8), Teeth(8,5), Lips(5,3)
    close_6h = df_6h['close'].values
    jaw = pd.Series(close_6h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close_6h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close_6h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 12h 20-period high and low for structure filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    high_20_12h_aligned = align_htf_to_ltf(prices, df_12h, high_20_12h)
    low_20_12h_aligned = align_htf_to_ltf(prices, df_12h, low_20_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (13 for jaw + 8 shift = 21, plus buffer)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(high_20_12h_aligned[i]) or 
            np.isnan(low_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: price above all lines AND above 12h 20-period high (bullish structure)
            if (price > jaw_aligned[i] and price > teeth_aligned[i] and price > lips_aligned[i] and
                price > high_20_12h_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: price below all lines AND below 12h 20-period low (bearish structure)
            elif (price < jaw_aligned[i] and price < teeth_aligned[i] and price < lips_aligned[i] and
                  price < low_20_12h_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below teeth (middle line)
            if price < teeth_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above teeth (middle line)
            if price > teeth_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsAlligator_12hStructure"
timeframe = "6h"
leverage = 1.0