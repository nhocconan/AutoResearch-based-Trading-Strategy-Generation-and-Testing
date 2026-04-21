#!/usr/bin/env python3
"""
4h_Donchian_20_Volume_TrendFilter
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA trend filter. 
Long when price breaks above upper band with volume > 1.5x 20-period average and price > 1d EMA50.
Short when price breaks below lower band with volume confirmation and price < 1d EMA50.
Designed for 4h timeframe to target 25-40 trades/year with tight entry conditions.
Works in bull markets by capturing breakouts and in bear markets by capturing breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian channels"""
    upper = np.full_like(high, np.nan)
    lower = np.full_like(high, np.nan)
    
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.full_like(close, np.nan)
    if len(close) >= period:
        ema[period-1] = np.mean(close[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(close)):
            ema[i] = (close[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = calculate_ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels on 4h data
    high = prices['high'].values
    low = prices['low'].values
    upper, lower = calculate_donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if EMA not ready
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long: price breaks above upper band with volume and trend filter
            if not np.isnan(upper[i]) and price > upper[i] and volume_ok and price > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume and trend filter
            elif not np.isnan(lower[i]) and price < lower[i] and volume_ok and price < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower band or trend changes
            if not np.isnan(lower[i]) and price < lower[i] or price < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper band or trend changes
            if not np.isnan(upper[i]) and price > upper[i] or price > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_20_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0