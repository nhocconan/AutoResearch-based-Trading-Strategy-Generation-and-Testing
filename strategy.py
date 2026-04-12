#!/usr/bin/env python3
# 4h_1d_donchian_volume_v1
# Hypothesis: 4-hour strategy using 1-day Donchian(20) breakout for entries, confirmed by volume > 1.5x 20-period average.
# Trend filter: price above/below 1-day EMA50 to align with higher timeframe direction.
# Works in bull/bear by requiring trend alignment and volume confirmation to avoid false breakouts.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

name = "4h_1d_donchian_volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day EMA50 for trend direction
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Previous 1-day Donchian(20) channels
    # Highest high of previous 20 days
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lowest low of previous 20 days
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous day's channel (breakout of prior 20-day range)
    donchian_high = np.roll(high_20, 1)
    donchian_low = np.roll(low_20, 1)
    
    # Align to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price above EMA50 (uptrend) AND breakout above Donchian high with volume
        if (close[i] > ema50_1d_aligned[i] and close[i] > donchian_high_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price below EMA50 (downtrend) AND breakdown below Donchian low with volume
        elif (close[i] < ema50_1d_aligned[i] and close[i] < donchian_low_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or price crosses back to opposite Donchian level
        elif position == 1 and close[i] < donchian_low_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donchian_high_aligned[i]:
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