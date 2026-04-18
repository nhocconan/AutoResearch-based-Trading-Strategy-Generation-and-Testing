#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R with 12h EMA trend filter and volume confirmation.
- Long: Williams %R crosses above -80, price > 12h EMA(34), volume > 1.5x average
- Short: Williams %R crosses below -20, price < 12h EMA(34), volume > 1.5x average
- Exit: Williams %R crosses back through -50 or volume < average
- Uses Williams %R for mean reversion in extremes, EMA for trend filter, volume for confirmation.
Designed for 20-50 trades/year to minimize fee drag and work in both bull/bear markets.
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
    ema[0] = close[0]
    for i in range(1, len(close)):
        ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R (14-period) on 4h
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Calculate EMA (34-period) on 12h
    ema_34_12h = calculate_ema(close_12h, 34)
    
    # Align EMA to 4h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need Williams %R, EMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Williams %R cross signals
        wr_cross_up = williams_r[i] > -80 and williams_r[i-1] <= -80
        wr_cross_down = williams_r[i] < -20 and williams_r[i-1] >= -20
        wr_cross_down_50 = williams_r[i] < -50 and williams_r[i-1] >= -50
        wr_cross_up_50 = williams_r[i] > -50 and williams_r[i-1] <= -50
        
        if position == 0:
            # Long: Williams %R crosses above -80, price > 12h EMA, volume confirmation
            if wr_cross_up and close[i] > ema_34_12h_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20, price < 12h EMA, volume confirmation
            elif wr_cross_down and close[i] < ema_34_12h_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses below -50 or volume drops below average
            if wr_cross_down_50 or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses above -50 or volume drops below average
            if wr_cross_up_50 or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0