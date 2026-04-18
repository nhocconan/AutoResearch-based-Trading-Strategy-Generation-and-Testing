#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA(34) trend filter.
- Long: Close > Donchian Upper(20) + Volume > 1.5x 20-period average + Close > 12h EMA(34)
- Short: Close < Donchian Lower(20) + Volume > 1.5x 20-period average + Close < 12h EMA(34)
- Exit: Opposite Donchian break (Lower for long exit, Upper for short exit)
- Uses 12h EMA(34) as trend filter to avoid counter-trend trades.
Designed for 20-50 trades/year on 4h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian Channels: upper = max(high, period), lower = min(low, period)."""
    upper = np.full(len(high), np.nan)
    lower = np.full(len(low), np.nan)
    
    for i in range(period - 1, len(high)):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    ema = np.full(len(close), np.nan)
    if len(close) < period:
        return ema
    ema[period - 1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = (close[i] * 2 / (period + 1)) + ema[i - 1] * (1 - 2 / (period + 1))
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA(34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(34) on 12h
    ema_34_12h = calculate_ema(close_12h, 34)
    
    # Align to 4h timeframe
    ema_34_12h_4h = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Donchian Channels (20-period) on 4h
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i - 20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian and volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_4h[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian Upper, volume confirmation, uptrend (price > 12h EMA)
            if close[i] > donchian_upper[i] and vol_confirmed and close[i] > ema_34_12h_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian Lower, volume confirmation, downtrend (price < 12h EMA)
            elif close[i] < donchian_lower[i] and vol_confirmed and close[i] < ema_34_12h_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian Lower
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian Upper
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_12hEMA34"
timeframe = "4h"
leverage = 1.0