#!/usr/bin/env python3
# 6h_1w_donchian_pivot_breakout_v1
# Strategy: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Weekly pivot points establish key support/resistance levels. 
# Price breaking above weekly R1 with Donchian breakout and volume confirms bullish momentum.
# Price breaking below weekly S1 with Donchian breakout and volume confirms bearish momentum.
# Works in both bull and bear markets by trading breakouts in direction of weekly pivot bias.
# Low frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_donchian_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar to close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Donchian channel (20-period) on 6h data
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper[i]  # Price breaks above Donchian upper
        breakout_down = close[i] < donchian_lower[i]  # Price breaks below Donchian lower
        
        # Weekly pivot bias conditions
        above_r1 = close[i] > r1_1w_aligned[i]  # Price above weekly R1
        below_s1 = close[i] < s1_1w_aligned[i]  # Price below weekly S1
        
        # Entry logic: Donchian breakout in direction of weekly pivot bias + volume
        if breakout_up and above_r1 and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and below_s1 and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price returns to weekly pivot level or opposite Donchian breakout
        elif position == 1 and (close[i] <= pivot_1w_aligned[i] or breakout_down):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= pivot_1w_aligned[i] or breakout_up):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals