#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter with 1-week Donchian breakout
# Long when price breaks above weekly Donchian(20) high AND weekly Choppiness > 61.8 (range regime)
# Short when price breaks below weekly Donchian(20) low AND weekly Choppiness > 61.8 (range regime)
# Exit when price crosses back inside the weekly Donchian channel
# Uses choppiness to identify range-bound markets where mean-reversion at channel extremes works
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop for Donchian and Choppiness
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Calculate weekly Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(highest high - lowest low over 14)))
    tr1 = high_1w[1:] - low_1w[:-1]
    tr2 = np.abs(high_1w[1:] - np.roll(close_1w, 1)[1:])
    tr3 = np.abs(low_1w[1:] - np.roll(close_1w, 1)[1:])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr14 * 14 / (highest_high - lowest_low)) / np.log10(10)
    chop_raw = np.where((highest_high - lowest_low) > 0, chop_raw, 50)  # avoid div/0
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_raw)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        chop = chop_aligned[i]
        
        # Only trade in range regime (Choppiness > 61.8)
        if chop <= 61.8:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: break below Donchian low (mean reentry in range)
            if price < low_20_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short setup: break above Donchian high (mean reentry in range)
            elif price > high_20_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above Donchian high (range mean reversion complete)
            if price > high_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below Donchian low (range mean reversion complete)
            if price < low_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Chop_1wDonchian_MeanReversion"
timeframe = "12h"
leverage = 1.0