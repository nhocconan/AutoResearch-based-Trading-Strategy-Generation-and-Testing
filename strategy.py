#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R with 1d trend filter and volume confirmation.
Williams %R identifies overbought/oversold conditions. In bull markets, buy when %R crosses above -80 from below with price above 1d EMA(50). In bear markets, sell when %R crosses below -20 from above with price below 1d EMA(50). Volume confirms momentum. Designed for 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R."""
    if len(close) < period:
        return np.full(len(close), np.nan)
    
    highest_high = np.full(len(close), np.nan)
    lowest_low = np.full(len(close), np.nan)
    
    for i in range(period-1, len(close)):
        highest_high[i] = np.max(high[i-(period-1):i+1])
        lowest_low[i] = np.min(low[i-(period-1):i+1])
    
    williams_r = np.full(len(close), np.nan)
    for i in range(period-1, len(close)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
        else:
            williams_r[i] = -50  # neutral when no range
    
    return williams_r

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    ema = np.full(len(close), np.nan)
    if len(close) < period:
        return ema
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = (close[i] * 2 / (period + 1)) + ema[i-1] * (1 - 2 / (period + 1))
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d
    ema_50_1d = calculate_ema(close_1d, 50)
    
    # Align to 4h timeframe
    ema_50_1d_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R on 4h
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Williams %R and volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_4h[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Williams %R signals
        williams_oversold = williams_r[i] <= -80
        williams_overbought = williams_r[i] >= -20
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below, price above EMA(50), volume confirmation
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_50_1d_4h[i] and vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above, price below EMA(50), volume confirmation
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_50_1d_4h[i] and vol_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R becomes overbought or price crosses below EMA(50)
            if williams_r[i] >= -20 or close[i] <= ema_50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R becomes oversold or price crosses above EMA(50)
            if williams_r[i] <= -80 or close[i] >= ema_50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0