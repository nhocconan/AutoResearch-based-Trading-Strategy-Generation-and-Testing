#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 12h EMA(34) trend filter and volume confirmation.
Williams %R identifies overbought/oversold conditions; EMA(34) defines trend direction.
Long when %R < -80 (oversold) and price > EMA(34) (uptrend); short when %R > -20 (overbought) and price < EMA(34) (downtrend).
Volume filter ensures momentum confirmation. Designed for 15-25 trades/year to minimize fee drag.
Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
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
        if highest_high[i] - lowest_low[i] != 0:
            williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
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
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R and EMA(34)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R(14) on 12h
    williams_r_12h = calculate_williams_r(high_12h, low_12h, close_12h, 14)
    
    # Calculate EMA(34) on 12h
    ema_34_12h = calculate_ema(close_12h, 34)
    
    # Align to 6h timeframe
    williams_r_12h_6h = align_htf_to_ltf(prices, df_12h, williams_r_12h)
    ema_34_12h_6h = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_12h_6h[i]) or np.isnan(ema_34_12h_6h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80), price above EMA(34) (uptrend), volume confirmation
            if williams_r_12h_6h[i] < -80 and close[i] > ema_34_12h_6h[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), price below EMA(34) (downtrend), volume confirmation
            elif williams_r_12h_6h[i] > -20 and close[i] < ema_34_12h_6h[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 (momentum fading) or price crosses below EMA(34)
            if williams_r_12h_6h[i] > -50 or close[i] <= ema_34_12h_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 (momentum fading) or price crosses above EMA(34)
            if williams_r_12h_6h[i] < -50 or close[i] >= ema_34_12h_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_12hEMA34_Volume"
timeframe = "6h"
leverage = 1.0