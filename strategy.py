#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams %R reversal with 1-day EMA trend filter and volume confirmation.
- Long: Williams %R < -80 (oversold), price > 1-day EMA50, volume > 1.3x average
- Short: Williams %R > -20 (overbought), price < 1-day EMA50, volume > 1.3x average
- Exit: Williams %R crosses above -50 (long) or below -50 (short)
- Williams %R identifies overextended moves; EMA50 filters trend direction; volume confirms conviction.
Designed for 20-40 trades/year (80-160 total) to minimize fee drag while capturing mean reversion in trends.
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
            williams_r[i] = -50  # avoid division by zero
    
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
    
    # Get 1-day data for EMA and Williams %R
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) on 1-day
    williams_r_14_1d = calculate_williams_r(high_1d, low_1d, close_1d, 14)
    
    # Calculate EMA (50-period) on 1-day
    ema_50_1d = calculate_ema(close_1d, 50)
    
    # Align to 4-hour timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_14_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume moving average (20-period) on 4-hour
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need Williams %R, EMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), price > EMA50, volume confirmation
            if williams_r_aligned[i] < -80 and close[i] > ema_50_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), price < EMA50, volume confirmation
            elif williams_r_aligned[i] > -20 and close[i] < ema_50_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0