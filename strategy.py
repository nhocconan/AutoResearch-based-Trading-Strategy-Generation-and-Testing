#!/usr/bin/env python3
"""
12h_1d_Donchian_Breakout
Hypothesis: Uses 12h Donchian channel breakout with 1d trend filter (EMA50) and volume confirmation.
In bull markets (price > EMA50), takes long breakouts above upper band.
In bear markets (price < EMA50), takes short breakdowns below lower band.
Volume confirmation ensures breakouts have institutional participation.
Works in both bull and bear markets by adapting to trend direction.
Target: 20-40 trades/year on 12h (80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channel on 12h (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Align daily EMA50 to 12h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: bullish if price > EMA50, bearish if price < EMA50
        bullish_trend = close[i] > ema_50_aligned[i]
        bearish_trend = close[i] < ema_50_aligned[i]
        
        # Long breakout: price breaks above upper Donchian band with volume expansion in bullish trend
        long_breakout = (high[i] > high_max_20[i]) and volume_expansion[i] and bullish_trend
        
        # Short breakdown: price breaks below lower Donchian band with volume expansion in bearish trend
        short_breakdown = (low[i] < low_min_20[i]) and volume_expansion[i] and bearish_trend
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif long_breakout and position == 1:
            signals[i] = position_size
        elif short_breakdown and position != -1:
            position = -1
            signals[i] = -position_size
        elif short_breakdown and position == -1:
            signals[i] = -position_size
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1d_Donchian_Breakout"
timeframe = "12h"
leverage = 1.0