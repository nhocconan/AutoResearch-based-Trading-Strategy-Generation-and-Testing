#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Williams %R for trend and 1d Williams %R for momentum.
# Williams %R on 12h determines the primary trend direction (overbought/oversold levels).
# Williams %R on 1d provides momentum signals for entry timing.
# Strategy goes long when 12h is oversold and 1d shows bullish momentum.
# Strategy goes short when 12h is overbought and 1d shows bearish momentum.
# Designed to work in both bull and bear markets by using 12h Williams %R to identify
# overextended conditions that are likely to reverse, while using 1d Williams %R
# to time entries with momentum confirmation.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 12h data ONCE for Williams %R trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on 12h data (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    wr_12h = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    
    # Align 12h Williams %R to 6h timeframe
    wr_12h_aligned = align_htf_to_ltf(prices, df_12h, wr_12h)
    
    # Load 1d data ONCE for Williams %R momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on 1d data (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    wr_1d = -100 * (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d)
    
    # Align 1d Williams %R to 6h timeframe
    wr_1d_aligned = align_htf_to_ltf(prices, df_1d, wr_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 14
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr_12h_aligned[i]) or 
            np.isnan(wr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 12h oversold (< -80) AND 1d showing bullish momentum (> -50 and rising)
            if (wr_12h_aligned[i] < -80 and 
                wr_1d_aligned[i] > -50 and 
                i > start and wr_1d_aligned[i] > wr_1d_aligned[i-1]):
                position = 1
                signals[i] = position_size
            # Short: 12h overbought (> -20) AND 1d showing bearish momentum (< -50 and falling)
            elif (wr_12h_aligned[i] > -20 and 
                  wr_1d_aligned[i] < -50 and 
                  i > start and wr_1d_aligned[i] < wr_1d_aligned[i-1]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: 12h becomes overbought OR 1d loses bullish momentum
            if (wr_12h_aligned[i] > -20 or 
                wr_1d_aligned[i] < -50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: 12h becomes oversold OR 1d loses bearish momentum
            if (wr_12h_aligned[i] < -80 or 
                wr_1d_aligned[i] > -50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h1d_WilliamsR_Momentum_v1"
timeframe = "6h"
leverage = 1.0