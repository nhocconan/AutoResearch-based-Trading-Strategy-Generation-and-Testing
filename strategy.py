#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 12h EMA Trend Filter and Volume Spike
- Uses Williams %R(14) from 6h timeframe for overbought/oversold signals
- 12h EMA50 defines higher timeframe trend filter: only trade counter-trend extreme reversals
- Volume confirmation (> 1.8x 20-period average) filters weak signals
- Exit when Williams %R returns to neutral zone (-50) or trend reverses
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by fading extremes in direction of higher timeframe trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # for EMA50, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price above 12h EMA50 AND volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price below 12h EMA50 AND volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral (-50) OR trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long when Williams %R >= -50 OR price closes below 12h EMA50
                if (williams_r[i] >= -50 or close[i] < ema_50_12h_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R <= -50 OR price closes above 12h EMA50
                if (williams_r[i] <= -50 or close[i] > ema_50_12h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0