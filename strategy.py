#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume spike confirmation.
- Williams %R(14) identifies overbought/oversold conditions
- 1d EMA50 defines primary trend (long when price > EMA50, short when price < EMA50)
- Volume spike >2.0x average confirms institutional participation
- Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Works in both bull and bear markets via trend filter
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
    
    # Williams %R(14) - primary signal
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Volume MA, Williams %R, EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Williams %R reversal signals
        williams_oversold = williams_r[i] < -80  # Oversold condition
        williams_overbought = williams_r[i] > -20  # Overbought condition
        williams_exit_oversold = williams_r[i] > -50  # Exit oversold
        williams_exit_overbought = williams_r[i] < -50  # Exit overbought
        
        # 1d EMA50 trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold reversal AND uptrend AND volume confirmation
            if williams_oversold and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought reversal AND downtrend AND volume confirmation
            elif williams_overbought and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R exits oversold OR trend breaks
            if williams_exit_oversold or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R exits overbought OR trend breaks
            if williams_exit_overbought or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Reversal_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0