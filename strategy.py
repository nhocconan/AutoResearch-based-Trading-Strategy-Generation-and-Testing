#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R with 1d EMA(34) trend filter and volume confirmation.
In bull markets: Buy when Williams %R shows oversold (below -80) while price is above 1d EMA(34) with volume confirmation.
In bear markets: Sell when Williams %R shows overbought (above -20) while price is below 1d EMA(34) with volume confirmation.
Williams %R provides mean-reversion signals within the trend, reducing whipsaw. Weekly volatility filter avoids high-risk periods.
Designed for 15-25 trades/year to minimize fee drag.
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
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50
    
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
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d
    ema_34_1d = calculate_ema(close_1d, 34)
    
    # Calculate Williams %R on 12h
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Align EMA(34) from 1d to 12h timeframe
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1d_12h[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80), price above EMA(34), volume confirmation
            if williams_r[i] < -80 and close[i] > ema_34_1d_12h[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), price below EMA(34), volume confirmation
            elif williams_r[i] > -20 and close[i] < ema_34_1d_12h[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50 or price crosses below EMA(34)
            if williams_r[i] > -50 or close[i] <= ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50 or price crosses above EMA(34)
            if williams_r[i] < -50 or close[i] >= ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0