#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1w EMA50 Trend + Volume Spike Confirmation
- Williams %R(14) identifies overbought/oversold conditions (< -80 for long, > -20 for short)
- 1w EMA(50) defines major trend direction (only long when price > EMA, short when price < EMA)
- Volume confirmation (> 1.8x 20-period average) ensures breakout momentum
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 1w trend and mean-reverting extremes
- Higher timeframe (1w) trend filter reduces whipsaw in sideways markets
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
    
    # Calculate weekly EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # Weekly EMA, Williams %R, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R conditions
        oversold = williams_r[i] < -80  # Extreme oversold
        overbought = williams_r[i] > -20  # Extreme overbought
        
        # Trend filter: price > EMA for uptrend, price < EMA for downtrend
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long conditions: Williams %R oversold, uptrend, volume spike
            long_signal = (oversold and 
                          uptrend and
                          volume[i] > 1.8 * vol_ma[i])
            
            # Short conditions: Williams %R overbought, downtrend, volume spike
            short_signal = (overbought and 
                           downtrend and
                           volume[i] > 1.8 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to neutral zone or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R rises above -50 (neutral) or trend turns down
                if (williams_r[i] > -50 or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R falls below -50 (neutral) or trend turns up
                if (williams_r[i] < -50 or 
                    not downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1wEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0