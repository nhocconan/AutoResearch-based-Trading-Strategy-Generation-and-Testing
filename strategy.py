#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R(14) with 1d EMA(50) trend filter and volume confirmation.
- Long: Williams %R crosses above -20, price > 1d EMA50, volume > 1.5x 20-period average
- Short: Williams %R crosses below -80, price < 1d EMA50, volume > 1.5x 20-period average
- Exit: opposite Williams %R threshold (-80 for long, -20 for short)
- Williams %R identifies overbought/oversold conditions; EMA50 filters trend direction.
Designed for 12-37 trades/year (50-150 total) to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R."""
    if len(high) < period:
        return np.full(len(high), np.nan)
    
    highest_high = np.full(len(high), np.nan)
    lowest_low = np.full(len(high), np.nan)
    
    for i in range(period - 1, len(high)):
        highest_high[i] = np.max(high[i - period + 1:i + 1])
        lowest_low[i] = np.min(low[i - period + 1:i + 1])
    
    williams_r = np.full(len(high), np.nan)
    for i in range(period - 1, len(high)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
    
    return williams_r

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    if len(close) < period:
        return np.full(len(close), np.nan)
    
    ema = np.full(len(close), np.nan)
    multiplier = 2 / (period + 1)
    ema[period - 1] = np.mean(close[:period])
    
    for i in range(period, len(close)):
        ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d
    ema_50_1d = calculate_ema(close_1d, 50)
    
    # Align to 6h timeframe
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 6h
    williams_r_14 = calculate_williams_r(high, low, close, 14)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need Williams %R, EMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_14[i]) or np.isnan(ema_50_1d_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -20, price > EMA50, volume confirmation
            if williams_r_14[i] > -20 and williams_r_14[i-1] <= -20 and close[i] > ema_50_1d_6h[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80, price < EMA50, volume confirmation
            elif williams_r_14[i] < -80 and williams_r_14[i-1] >= -80 and close[i] < ema_50_1d_6h[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses below -80
            if williams_r_14[i] < -80 and williams_r_14[i-1] >= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses above -20
            if williams_r_14[i] > -20 and williams_r_14[i-1] <= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR14_EMA50_Volume"
timeframe = "6h"
leverage = 1.0