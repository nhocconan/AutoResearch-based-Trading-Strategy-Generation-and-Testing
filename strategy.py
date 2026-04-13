#!/usr/bin/env python3
"""
1d_1w_Trend_Following_with_Volume_Filter
Hypothesis: Weekly trend direction (using 20-period EMA) filters daily breakouts at 10-period high/low with volume confirmation.
Captures medium-term trends while avoiding counter-trend trades. Works in bull markets via long breakouts above weekly EMA,
and in bear markets via short breakouts below weekly EMA. Volume filter reduces false breakouts. Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly 20-period EMA for trend filter
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Daily 10-period high/low for breakout levels
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (vol_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_20_aligned[i]) or np.isnan(high_10[i]) or 
            np.isnan(low_10[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above 10-day high, above weekly EMA, with volume expansion
        long_breakout = (close[i] > high_10[i]) and (close[i] > ema_20_aligned[i]) and volume_expansion[i]
        
        # Short breakdown: price breaks below 10-day low, below weekly EMA, with volume expansion
        short_breakout = (close[i] < low_10[i]) and (close[i] < ema_20_aligned[i]) and volume_expansion[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_Trend_Following_with_Volume_Filter"
timeframe = "1d"
leverage = 1.0