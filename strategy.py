#!/usr/bin/env python3
"""
1h_4h_1d_Momentum_Pullback_v3
Hypothesis: Combines 4h trend direction (EMA21) with 1d pullback to EMA50 and volume confirmation on 1h.
In trending markets (4h EMA21 slope), enters long when price pulls back to 1d EMA50 with volume expansion.
Short when 4h trend is down and price pulls back to 1d EMA50 from above.
Uses 1h for entry timing precision, 4h for trend direction, 1d for pullback level.
Targets 15-35 trades/year by requiring trend alignment, pullback, and volume expansion.
Works in bull markets via long pullsbacks and bear markets via short pullbacks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4h EMA21 for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_prev = np.roll(ema_4h, 1)
    ema_4h_prev[0] = ema_4h[0]
    ema_4h_slope = ema_4h - ema_4h_prev  # Positive = uptrend, Negative = downtrend
    
    # Get 1d data for pullback level
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for pullback level
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF data to 1h timeframe
    ema_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_slope)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h data for entry timing and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% of capital
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_4h_slope_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 4h EMA21 slope
        trend_up = ema_4h_slope_aligned[i] > 0
        trend_down = ema_4h_slope_aligned[i] < 0
        
        # Pullback condition: price near 1d EMA50 (within 0.5%)
        pullback_level = ema_50_1d_aligned[i]
        near_pullback = abs(close[i] - pullback_level) / pullback_level < 0.005
        
        # Entry conditions
        long_entry = trend_up and near_pullback and volume_expansion[i]
        short_entry = trend_down and near_pullback and volume_expansion[i]
        
        # Exit conditions: reverse signal or loss of momentum
        long_exit = not trend_up or (close[i] > pullback_level * 1.015)  # Take profit at 1.5% above
        short_exit = not trend_down or (close[i] < pullback_level * 0.985)  # Take profit at 1.5% below
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_Momentum_Pullback_v3"
timeframe = "1h"
leverage = 1.0