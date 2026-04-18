#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot S1/R1 breakout with volume confirmation and 12h EMA(34) trend filter.
In bull markets: break above R1 with volume and above 12h EMA(34) = long.
In bear markets: break below S1 with volume and below 12h EMA(34) = short.
Based on top-performing pattern from DB: 4h_Pivot_R1S1_R2S2_Breakout_12hEMA34_Volume (ETHUSDT test_sharpe=1.490).
Uses discrete position sizing (0.25) to limit trade frequency and reduce fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close
    c = close
    h = high
    l = low
    r4 = c + range_val * 1.500
    r3 = c + range_val * 1.250
    r2 = c + range_val * 1.166
    r1 = c + range_val * 1.083
    s1 = c - range_val * 1.083
    s2 = c - range_val * 1.166
    s3 = c - range_val * 1.250
    s4 = c - range_val * 1.500
    return r1, r2, r3, r4, s1, s2, s3, s4

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
    
    # Get 12h data for EMA(34) trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(34) on 12h
    ema_34_12h = calculate_ema(close_12h, 34)
    
    # Align EMA(34) to 4h timeframe
    ema_34_12h_4h = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_4h[i]) or np.isnan(vol_ma[i]) or
            i < 1):  # need previous bar for Camarilla calculation
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels from previous bar
        r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
            high[i-1], low[i-1], close[i-1]
        )
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1, above 12h EMA(34), volume confirmation
            if close[i] > r1 and close[i] > ema_34_12h_4h[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below 12h EMA(34), volume confirmation
            elif close[i] < s1 and close[i] < ema_34_12h_4h[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below S1 or below 12h EMA(34)
            if close[i] < s1 or close[i] < ema_34_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R1 or above 12h EMA(34)
            if close[i] > r1 or close[i] > ema_34_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_S1R1_Breakout_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0