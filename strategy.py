#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme reversal with 1d EMA200 trend filter and volume spike confirmation
- Uses Williams %R(14) from 6h timeframe for overextension signals: long when %R < -80, short when %R > -20
- 1d EMA200 defines higher timeframe trend: only trade reversals in trend direction (pullbacks to EMA200)
- Volume confirmation (> 2.0x 20-period average) filters false reversals
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 1d trend
- Williams %R provides mean-reversion signals in ranging markets and exhaustion signals in trends
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
    if n < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: > 2.0x 20-period average (tight to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(200, 14, 20)  # for EMA200, Williams %R, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80) with 1d uptrend and volume spike
            long_setup = (williams_r[i] < -80 and 
                         close[i] > ema_200_1d_aligned[i] and
                         volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: Williams %R overbought (> -20) with 1d downtrend and volume spike
            short_setup = (williams_r[i] > -20 and 
                          close[i] < ema_200_1d_aligned[i] and
                          volume[i] > 2.0 * vol_ma[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to neutral range (-50) or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns above -50 or price closes below 1d EMA200
                if (williams_r[i] > -50 or 
                    close[i] < ema_200_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R returns below -50 or price closes above 1d EMA200
                if (williams_r[i] < -50 or 
                    close[i] > ema_200_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA200_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0