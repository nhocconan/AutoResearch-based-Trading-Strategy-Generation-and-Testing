#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 12h EMA50 trend filter and volume confirmation
- Williams %R(14) identifies overbought/oversold conditions (< -80 for long, > -20 for short)
- 12h EMA(50) defines trend direction (only long when price > EMA, short when price < EMA)
- Volume confirmation (> 1.5x 20-period average) filters low-momentum signals
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 12h trend
- Williams %R extremes work well in ranging markets which occur in both bull/bear regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams %R(14)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Calculate 12h EMA(50) for trend filter
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80  # Long signal
        overbought = williams_r_aligned[i] > -20  # Short signal
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long conditions: Williams %R oversold, uptrend, volume spike
            long_signal = (oversold and 
                          uptrend and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: Williams %R overbought, downtrend, volume spike
            short_signal = (overbought and 
                           downtrend and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Williams %R extreme or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R becomes overbought OR trend turns down
                if (overbought or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R becomes oversold OR trend turns up
                if (oversold or 
                    not downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0