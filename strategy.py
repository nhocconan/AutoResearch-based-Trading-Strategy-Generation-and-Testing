#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1d EMA50 Trend and Volume Confirmation
- Williams %R(14) identifies overbought/oversold conditions (< -80 for long, > -20 for short)
- 1d EMA(50) ensures alignment with daily trend for multi-timeframe confirmation
- Volume > 1.8x 20-period average confirms momentum behind the move
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in both bull and bear markets by fading extremes in direction of 1d trend
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R(14) on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 1.8x 20-period average on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # EMA1d, Williams %R, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R extreme signals with trend filter and volume confirmation
        # Long: Williams %R < -80 (oversold) + uptrend + volume confirmation
        # Short: Williams %R > -20 (overbought) + downtrend + volume confirmation
        long_signal = (williams_r[i] < -80 and 
                      close[i] > ema_50_1d_aligned[i] and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (williams_r[i] > -20 and 
                       close[i] < ema_50_1d_aligned[i] and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or Williams %R returns to neutral territory
            exit_signal = False
            
            if position == 1:
                # Exit long: trend reversal or Williams %R > -50 (exiting oversold)
                if (close[i] < ema_50_1d_aligned[i] or 
                    williams_r[i] > -50):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend reversal or Williams %R < -50 (exiting overbought)
                if (close[i] > ema_50_1d_aligned[i] or 
                    williams_r[i] < -50):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0