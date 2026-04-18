#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian(20) breakout with 1-week EMA trend filter and volume confirmation.
- Long: Price breaks above 1-day Donchian upper band, 1-week EMA(20) > price (bullish bias), volume > 1.5x 20-day average
- Short: Price breaks below 1-day Donchian lower band, 1-week EMA(20) < price (bearish bias), volume > 1.5x 20-day average
- Exit: Opposite Donchian band touch
- Uses 1-week EMA for trend bias to avoid counter-trend trades in opposing markets.
Designed for 7-25 trades/year (30-100 total) to minimize fee drag.
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
        ema[i] = (values[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian bands
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 1d
    donchian_high = np.full(len(high_1d), np.nan)
    donchian_low = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):  # 20-period lookback
        donchian_high[i] = np.max(high_1d[i-19:i+1])
        donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate EMA (20-period) on 1w
    ema_20_1w = calculate_ema(close_1w, 20)
    
    # Align to 1d timeframe
    donchian_high_1d = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_1d = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_20_1w_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need Donchian, EMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or 
            np.isnan(ema_20_1w_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, EMA > price (bullish), volume confirmation
            if close[i] > donchian_high_1d[i] and ema_20_1w_1d[i] > close[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, EMA < price (bearish), volume confirmation
            elif close[i] < donchian_low_1d[i] and ema_20_1w_1d[i] < close[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches Donchian low
            if close[i] <= donchian_low_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches Donchian high
            if close[i] >= donchian_high_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA20_Volume"
timeframe = "1d"
leverage = 1.0