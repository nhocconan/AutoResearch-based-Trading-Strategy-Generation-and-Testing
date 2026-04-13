#!/usr/bin/env python3
"""
1d_Weekly_Momentum_With_Volume_Confirmation
Hypothesis: On daily timeframe, buy when weekly momentum (price change over 4 weeks) is positive and volume > 2x 20-day average, sell when weekly momentum is negative with volume confirmation. Uses weekly trend filter to capture momentum in both bull and bear markets, with volume confirmation to avoid false signals. Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag. Weekly momentum captures intermediate-term trends, and volume ensures institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 2x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 2.0)
    
    # Get weekly data for momentum calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need at least 5 weeks for 4-week momentum
        return np.zeros(n)
    
    # Weekly close prices for momentum (4-week change)
    weekly_close = df_1w['close'].values
    weekly_momentum = np.zeros(len(weekly_close))
    # Calculate 4-week momentum: (current weekly close - close 4 weeks ago) / close 4 weeks ago
    for i in range(4, len(weekly_close)):
        if weekly_close[i-4] != 0:
            weekly_momentum[i] = (weekly_close[i] - weekly_close[i-4]) / weekly_close[i-4]
        else:
            weekly_momentum[i] = 0
    
    # Align weekly momentum to daily timeframe (wait for weekly close)
    weekly_momentum_aligned = align_htf_to_ltf(prices, df_1w, weekly_momentum)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(60, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(weekly_momentum_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: positive weekly momentum with volume expansion
        long_signal = (weekly_momentum_aligned[i] > 0 and volume_expansion[i])
        
        # Short signal: negative weekly momentum with volume expansion
        short_signal = (weekly_momentum_aligned[i] < 0 and volume_expansion[i])
        
        if position == 1 and short_signal:
            position = -1
            signals[i] = -position_size
        elif position == -1 and long_signal:
            position = 1
            signals[i] = position_size
        elif position == 0:
            if long_signal:
                position = 1
                signals[i] = position_size
            elif short_signal:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_Weekly_Momentum_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0