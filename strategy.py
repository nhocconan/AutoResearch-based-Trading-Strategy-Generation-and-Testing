#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter combined with 1-day Williams %R mean reversion
# We go long when CHOP > 61.8 (range regime) and Williams %R < -80 (oversold)
# We go short when CHOP > 61.8 (range regime) and Williams %R > -20 (overbought)
# This strategy targets range-bound markets which are common in BTC/ETH during bear markets
# Uses 4h timeframe to target 20-50 trades/year, avoiding excessive frequency
# Choppiness Index filters out trending markets where mean reversion fails
# Williams %R provides timely entry signals at extremes in range-bound conditions

name = "4h_ChopRange_WilliamsMR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data once for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on daily data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate Choppiness Index on 4h data
    # CHOP = 100 * log10(sum(ATR over n) / (max(high) - min(low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop = np.where((max_high - min_low) == 0, 50, chop)  # avoid div by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        williams_r_val = williams_r_aligned[i]
        chop_val = chop[i]
        
        if position == 0:
            # Enter long: range regime + Williams %R oversold
            if chop_val > 61.8 and williams_r_val < -80:
                signals[i] = 0.25
                position = 1
            # Enter short: range regime + Williams %R overbought
            elif chop_val > 61.8 and williams_r_val > -20:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral or regime changes
            if williams_r_val > -50 or chop_val <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral or regime changes
            if williams_r_val < -50 or chop_val <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals