#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R (14) + volume confirmation + 1w EMA20 trend filter.
- Long: Williams %R crosses above -80 (oversold reversal), volume > 1.5x average, price > 1w EMA20
- Short: Williams %R crosses below -20 (overbought reversal), volume > 1.5x average, price < 1w EMA20
- Exit: Williams %R crosses opposite threshold (-20 for long, -80 for short)
- Uses Williams %R for mean reversion in ranging markets, EMA20 for trend filter.
Designed for 7-25 trades/year (30-100 total) to minimize fee drag.
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
        else:
            williams_r[i] = -50  # Avoid division by zero
    
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
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) on 1d
    williams_r_14_1d = calculate_williams_r(high_1d, low_1d, close_1d, 14)
    
    # Get 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA (20-period) on 1w
    ema_20_1w = calculate_ema(close_1w, 20)
    
    # Align to 1d timeframe
    williams_r_14_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_14_1d)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need Williams %R, EMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_14_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80, volume confirmation, price > EMA20
            if (williams_r_14_1d_aligned[i] > -80 and 
                williams_r_14_1d_aligned[i-1] <= -80 and 
                vol_confirmed and 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20, volume confirmation, price < EMA20
            elif (williams_r_14_1d_aligned[i] < -20 and 
                  williams_r_14_1d_aligned[i-1] >= -20 and 
                  vol_confirmed and 
                  close[i] < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses below -20
            if williams_r_14_1d_aligned[i] < -20 and williams_r_14_1d_aligned[i-1] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses above -80
            if williams_r_14_1d_aligned[i] > -80 and williams_r_14_1d_aligned[i-1] <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR14_Volume_EMA20"
timeframe = "1d"
leverage = 1.0