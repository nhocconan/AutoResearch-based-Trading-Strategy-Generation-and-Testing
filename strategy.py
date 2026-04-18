#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R with 1d EMA34 filter and volume confirmation.
- Long: Williams %R < -80 (oversold), price > 1d EMA34, volume > 1.5x 20-period average
- Short: Williams %R > -20 (overbought), price < 1d EMA34, volume > 1.5x 20-period average
- Exit: Williams %R crosses above -50 (long) or below -50 (short)
- Uses Williams %R for mean reversion in ranging markets, EMA34 for trend filter.
Designed for 12-37 trades/year (50-150 total) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R."""
    if len(high) < period:
        return np.full(len(high), np.nan)
    
    highest_high = np.full(len(high), np.nan)
    lowest_low = np.full(len(low), np.nan)
    
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

def calculate_ema(values, period):
    """Calculate Exponential Moving Average."""
    if len(values) < period:
        return np.full(len(values), np.nan)
    
    ema = np.full(len(values), np.nan)
    multiplier = 2 / (period + 1)
    ema[period - 1] = np.mean(values[:period])
    
    for i in range(period, len(values)):
        ema[i] = (values[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) on 1d
    williams_r_14_1d = calculate_williams_r(high_1d, low_1d, close_1d, 14)
    
    # Calculate EMA (34-period) on 1d
    ema_34_1d = calculate_ema(close_1d, 34)
    
    # Align to 12h timeframe
    williams_r_14_1d_12h = align_htf_to_ltf(prices, df_1d, williams_r_14_1d)
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need Williams %R, EMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_14_1d_12h[i]) or np.isnan(ema_34_1d_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), price > EMA34, volume confirmation
            if williams_r_14_1d_12h[i] < -80 and close[i] > ema_34_1d_12h[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), price < EMA34, volume confirmation
            elif williams_r_14_1d_12h[i] > -20 and close[i] < ema_34_1d_12h[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50
            if williams_r_14_1d_12h[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50
            if williams_r_14_1d_12h[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR14_EMA34_Volume"
timeframe = "12h"
leverage = 1.0