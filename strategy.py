#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation.
- Long: price breaks above Donchian upper band, price > 12h EMA34, volume > 1.5x average
- Short: price breaks below Donchian lower band, price < 12h EMA34, volume > 1.5x average
- Exit: opposite Donchian band touch
- Uses 12h EMA for trend filter to avoid counter-trend trades in choppy markets.
Designed for 20-50 trades/year (80-200 total) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(values, period):
    """Calculate Exponential Moving Average."""
    if len(values) < period:
        return np.full(len(values), np.nan)
    
    ema = np.full(len(values), np.nan)
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(values[:period])
    
    for i in range(period, len(values)):
        ema[i] = (values[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h
    ema34_12h = calculate_ema(close_12h, 34)
    ema34_12h_4h = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need Donchian, EMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema34_12h_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, price > EMA34, volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema34_12h_4h[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, price < EMA34, volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema34_12h_4h[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches Donchian low
            if close[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches Donchian high
            if close[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_EMA34_Volume"
timeframe = "4h"
leverage = 1.0