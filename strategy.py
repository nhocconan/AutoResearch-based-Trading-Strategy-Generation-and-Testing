#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA(34) trend filter and volume confirmation.
Long when price breaks above Donchian high and 12h EMA(34) rising; short when price breaks below Donchian low and 12h EMA(34) falling.
Volume filter ensures breakouts are genuine. Designed for 20-40 trades/year to minimize fee drag.
Works in bull markets (breakouts catch trends) and bear markets (short breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA(34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(34) on 12h
    ema_34_12h = calculate_ema(close_12h, 34)
    
    # Align to 4h timeframe
    ema_34_12h_4h = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_4h[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, EMA rising, volume confirmation
            if (close[i] > donchian_high[i] and 
                ema_34_12h_4h[i] > ema_34_12h_4h[i-1] and vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, EMA falling, volume confirmation
            elif (close[i] < donchian_low[i] and 
                  ema_34_12h_4h[i] < ema_34_12h_4h[i-1] and vol_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or EMA starts falling
            if (close[i] < donchian_low[i] or 
                ema_34_12h_4h[i] < ema_34_12h_4h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or EMA starts rising
            if (close[i] > donchian_high[i] or 
                ema_34_12h_4h[i] > ema_34_12h_4h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0