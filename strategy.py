#!/usr/bin/env python3
# 1d_WeeklyTrend_Following
# Hypothesis: On daily timeframe, go long when price breaks above weekly Donchian high (20-period) with volume confirmation, short when breaks below weekly Donchian low. Uses weekly trend filter (price above/below weekly EMA20) to avoid counter-trend trades. Designed for low-frequency trades (10-25/year) to minimize fee drag, works in bull via trend continuation and bear via trend reversals at extremes.

name = "1d_WeeklyTrend_Following"
timeframe = "1d"
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
    
    # Load weekly data for trend and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate weekly Donchian channels (20-period)
    # Using rolling window on weekly data
    high_roll = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high = align_htf_to_ltf(prices, df_1w, high_roll)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_roll)
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        ema20_val = ema20_1w_aligned[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Price breaks above weekly Donchian high, above weekly EMA20 trend, with volume confirmation
            if close[i] > donchian_high_val and close[i] > ema20_val and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low, below weekly EMA20 trend, with volume confirmation
            elif close[i] < donchian_low_val and close[i] < ema20_val and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly Donchian low or below weekly EMA20
            if close[i] < donchian_low_val or close[i] < ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly Donchian high or above weekly EMA20
            if close[i] > donchian_high_val or close[i] > ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals