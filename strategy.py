#!/usr/bin/env python3
"""
1d_1w_donchian_breakout_volume - Trend following with weekly trend filter
Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation.
Works in bull markets (trend following) and bear markets (avoids counter-trend trades).
Targets 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
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
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period) using previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period high/low of previous day's data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Shift by 1 to use previous day's channels (no look-ahead)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    donchian_high[0] = np.nan
    donchian_low[0] = np.nan
    
    # Align Donchian levels to daily timeframe (already aligned, just copy)
    donchian_high_d = donchian_high
    donchian_low_d = donchian_low
    
    # Get weekly data for trend filter (EMA 21)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21 = np.full(len(close_1w), np.nan)
    
    # Calculate EMA 21
    alpha = 2.0 / (21 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_21[i] = close_1w[i]
        elif np.isnan(close_1w[i]):
            ema_21[i] = ema_21[i-1]
        else:
            ema_21[i] = alpha * close_1w[i] + (1 - alpha) * ema_21[i-1]
    
    # Align weekly EMA to daily timeframe
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_d[i]) or np.isnan(donchian_low_d[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Weekly trend filter: price above/below EMA 21
        uptrend = close[i] > ema_21_aligned[i]
        downtrend = close[i] < ema_21_aligned[i]
        
        # Entry conditions: Donchian breakout with volume and trend filter
        long_breakout = close[i] > donchian_high_d[i] and volume_filter and uptrend
        short_breakout = close[i] < donchian_low_d[i] and volume_filter and downtrend
        
        # Exit conditions: opposite Donchian level touch
        long_exit = close[i] < donchian_low_d[i]
        short_exit = close[i] > donchian_high_d[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_breakout_volume"
timeframe = "1d"
leverage = 1.0