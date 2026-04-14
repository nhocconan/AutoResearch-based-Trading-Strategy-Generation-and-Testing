#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter combined with 1-day Williams %R mean reversion
# Long when CHOPPINESS > 61.8 (range) AND Williams %R < -80 (oversold)
# Short when CHOPPINESS > 61.8 (range) AND Williams %R > -20 (overbought)
# Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# Uses 12h chart for entries, 1d for regime filter, targeting 50-150 trades over 4 years
# Works in both bull/bear markets by focusing on mean reversion in ranging conditions

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load daily data ONCE before loop for Williams %R and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R on daily: (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(df_1d['high']).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate Choppiness Index on daily: measures ranging vs trending
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(lookback)
    tr1 = pd.Series(df_1d['high']).rolling(window=1, min_periods=1).max() - pd.Series(df_1d['low']).rolling(window=1, min_periods=1).min()
    tr2 = abs(pd.Series(df_1d['high']).rolling(window=1, min_periods=1).max() - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']).rolling(window=1, min_periods=1).min() - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=lookback, min_periods=lookback).sum().values
    highest_high = pd.Series(df_1d['high']).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=lookback, min_periods=lookback).min().values
    chop = 100 * np.log10(atr / (highest_high - lowest_low)) / np.log10(lookback)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = lookback
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Long: ranging market + oversold
            if chop_val > 61.8 and wr < -80:
                position = 1
                signals[i] = position_size
            # Short: ranging market + overbought
            elif chop_val > 61.8 and wr > -20:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses back above -50
            if wr > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses back below -50
            if wr < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Chop_WilliamsR_MeanReversion"
timeframe = "12h"
leverage = 1.0