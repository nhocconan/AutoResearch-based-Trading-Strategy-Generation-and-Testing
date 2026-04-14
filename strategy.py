#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + Daily Trend Filter
Long when price breaks above Donchian(20) high with volume > 2x average and daily close > daily open.
Short when price breaks below Donchian(20) low with volume > 2x average and daily close < daily open.
Exit when price crosses the Donchian midpoint.
Designed for low turnover: ~20-40 trades/year per symbol.
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
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_open = df_1d['open'].values
    
    # Daily trend: 1 if bullish (close > open), -1 if bearish (close < open)
    daily_bullish = (daily_close > daily_open).astype(float)
    daily_bearish = (daily_close < daily_open).astype(float)
    
    # Align daily trend to 4h timeframe
    daily_bull_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bear_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned daily trend values
        daily_bull = daily_bull_aligned[i]
        daily_bear = daily_bear_aligned[i]
        
        if np.isnan(daily_bull) or np.isnan(daily_bear):
            continue
        
        if position == 0:
            # Long: Break above Donchian high, volume spike, daily bullish
            if close[i] > high_roll[i] and volume[i] > vol_ma[i] * 2.0 and daily_bull > 0.5:
                position = 1
                signals[i] = position_size
            # Short: Break below Donchian low, volume spike, daily bearish
            elif close[i] < low_roll[i] and volume[i] > vol_ma[i] * 2.0 and daily_bear > 0.5:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price crosses below Donchian midpoint
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price crosses above Donchian midpoint
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_Volume_DailyTrend"
timeframe = "4h"
leverage = 1.0