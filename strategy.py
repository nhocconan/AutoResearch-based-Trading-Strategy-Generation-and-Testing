#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Mean Reversion + 1d EMA50 Trend + Volume Confirmation
Williams %R identifies overbought/oversold conditions. In ranging markets, 
mean reversion from extreme %R levels with trend alignment and volume confirmation
provides edge. 12h timeframe reduces noise and trade frequency while capturing
significant moves. Target: 12-37 trades/year (50-150 over 4 years) with discrete sizing 0.25.
Works in both bull/bear via trend filter and mean reversion logic.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R on 12h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, period, 20)  # need EMA50_1d, Williams %R, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume confirmation
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.3 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume confirmation
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.3 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50) OR trend reversal
            exit_signal = False
            if position == 1:
                # Exit long when Williams %R > -50 (returns from oversold) OR price < 1d EMA50
                if williams_r[i] > -50 or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R < -50 (returns from overbought) OR price > 1d EMA50
                if williams_r[i] < -50 or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_MeanReversion_1dEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0