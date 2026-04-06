#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h momentum with 1w trend filter and volume confirmation
# Enter long when: 6h price > 6h EMA(20), price breaks above 1w Donchian high(10), volume > 2x 6h average
# Enter short when: 6h price < 6h EMA(20), price breaks below 1w Donchian low(10), volume > 2x 6h average
# Exit when: price crosses back below/above 6h EMA(20) OR opposite Donchian break occurs
# Uses weekly structure to capture major moves, targeting 50-150 trades over 4 years

name = "6h_momentum_1w_donchian_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h EMA(20) for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    
    # 1w Donchian channels (10-period for sensitivity)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high/low (10-period)
    donch_high = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    donch_low = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(ema_20[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price < 6h EMA(20) OR price < 1w Donchian low (reversal signal)
            if close[i] < ema_20[i] or close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price > 6h EMA(20) OR price > 1w Donchian high (reversal signal)
            if close[i] > ema_20[i] or close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: trend filter + Donchian break + volume
            if close[i] > ema_20[i] and close[i] > donch_high_aligned[i] and volume[i] > volume_threshold[i]:
                # Bullish: above EMA and breaking weekly high with volume
                signals[i] = 0.25
                position = 1
            elif close[i] < ema_20[i] and close[i] < donch_low_aligned[i] and volume[i] > volume_threshold[i]:
                # Bearish: below EMA and breaking weekly low with volume
                signals[i] = -0.25
                position = -1
    
    return signals