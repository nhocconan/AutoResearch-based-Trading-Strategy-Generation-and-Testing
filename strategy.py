#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout
Hypothesis: Breakouts above weekly Donchian(20) high in uptrend (price > 1d EMA50) and breakdowns below weekly Donchian low in downtrend (price < 1d EMA50) with volume confirmation (volume > 1.8x 20-period average). Uses weekly trend filter to avoid counter-trend trades. Designed for 6h timeframe to capture multi-day moves with low trade frequency.
"""

name = "6h_WeeklyPivot_Donchian_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_20w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h timeframe
    donchian_high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    donchian_low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Get daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    cooldown = 0  # cooldown counter to prevent immediate re-entry
    
    for i in range(50, n):
        # Decrease cooldown if active
        if cooldown > 0:
            cooldown -= 1
        
        if position == 0 and cooldown == 0:
            # LONG: Price breaks above weekly Donchian high with volume confirmation in uptrend
            if donchian_high_20w_aligned[i] > 0 and not np.isnan(donchian_high_20w_aligned[i]) and \
               high[i] > donchian_high_20w_aligned[i] and volume_confirmed[i] and \
               close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low with volume confirmation in downtrend
            elif donchian_low_20w_aligned[i] > 0 and not np.isnan(donchian_low_20w_aligned[i]) and \
                 low[i] < donchian_low_20w_aligned[i] and volume_confirmed[i] and \
                 close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below weekly Donchian high or trend weakens
            if donchian_high_20w_aligned[i] > 0 and not np.isnan(donchian_high_20w_aligned[i]) and \
               (low[i] < donchian_high_20w_aligned[i] or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                cooldown = 3  # 3-bar cooldown after exit
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above weekly Donchian low or trend weakens
            if donchian_low_20w_aligned[i] > 0 and not np.isnan(donchian_low_20w_aligned[i]) and \
               (high[i] > donchian_low_20w_aligned[i] or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                cooldown = 3  # 3-bar cooldown after exit
            else:
                signals[i] = -0.25
    
    return signals