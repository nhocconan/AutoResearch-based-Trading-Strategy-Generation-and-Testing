#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly EMA trend filter and volume confirmation
# Breakouts above 1-week high or below 1-week low capture strong momentum.
# Weekly EMA20 filters for major trend direction, avoiding counter-trend trades.
# Volume confirmation ensures institutional participation.
# Designed for 1d timeframe with 1h reference for signal timing precision.
# Targets 10-20 trades per year (~40-80 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by aligning with major trend via weekly EMA.

name = "1d_Donchian20_1wEMA20_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for signal timing precision
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # Get weekly data for Donchian channels and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling max/min for Donchian
    high_max = np.full_like(high_1w, np.nan)
    low_min = np.full_like(low_1w, np.nan)
    
    for i in range(20, len(high_1w)):
        high_max[i] = np.max(high_1w[i-20:i])
        low_min[i] = np.min(low_1w[i-20:i])
    
    # Align Donchian levels to 1h timeframe
    donchian_high_1h = align_htf_to_ltf(prices, df_1h, high_max)
    donchian_low_1h = align_htf_to_ltf(prices, df_1h, low_min)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20 = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20[i] = (close_1w[i] * 2/21) + (ema_20[i-1] * 19/21)
    
    # Align EMA to 1h timeframe
    ema_20_1h = align_htf_to_ltf(prices, df_1h, ema_20)
    
    # Volume confirmation using 1h volume
    vol_1h = df_1h['volume'].values
    vol_ma = np.full_like(vol_1h, np.nan)
    for i in range(20, len(vol_1h)):
        vol_ma[i] = np.mean(vol_1h[i-20:i])
    vol_ratio = np.where(vol_ma > 0, vol_1h / vol_ma, 1.0)
    vol_confirmed = vol_ratio > 1.5
    
    # Align volume confirmation to 1h timeframe (already in 1h)
    vol_confirmed_1h = vol_confirmed
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_1h[i]) or np.isnan(donchian_low_1h[i]) or 
            np.isnan(ema_20_1h[i]) or np.isnan(vol_confirmed_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high, above weekly EMA, volume confirmed
            if close[i] > donchian_high_1h[i] and close[i] > ema_20_1h[i] and vol_confirmed_1h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low, below weekly EMA, volume confirmed
            elif close[i] < donchian_low_1h[i] and close[i] < ema_20_1h[i] and vol_confirmed_1h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly Donchian low or below weekly EMA
            if close[i] < donchian_low_1h[i] or close[i] < ema_20_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly Donchian high or above weekly EMA
            if close[i] > donchian_high_1h[i] or close[i] > ema_20_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals