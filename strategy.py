#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeTrend_1dTrend
Hypothesis: Uses 4-hour Donchian channel (20-period) breakouts with volume confirmation (>1.5x average) and 1-day EMA50 trend filter. Trades in direction of higher timeframe trend to avoid counter-trend whipsaws. Designed for low trade frequency (15-25/year) to minimize fee drag while capturing strong trending moves in both bull and bear markets.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4-hour Donchian channels (20-period)
    # Donchian upper = highest high of last 20 periods
    # Donchian lower = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-period moving average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 and Donchian to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1-day EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_upper[i] and vol_confirm and uptrend
        short_breakout = close[i] < donchian_lower[i] and vol_confirm and downtrend
        
        # Exit conditions: opposite Donchian band touch
        long_exit = close[i] < donchian_lower[i]
        short_exit = close[i] > donchian_upper[i]
        
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeTrend_1dTrend"
timeframe = "4h"
leverage = 1.0