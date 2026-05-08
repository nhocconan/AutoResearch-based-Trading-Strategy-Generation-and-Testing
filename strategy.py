#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 1-week Donchian upper (highest high of last 20 weeks) AND 1d close > 1-week EMA50 AND volume > 1.5x 20-day average.
# Short when price breaks below 1-week Donchian lower (lowest low of last 20 weeks) AND 1d close < 1-week EMA50 AND volume > 1.5x 20-day average.
# Exit when price crosses back inside the weekly Donchian channel.
# Weekly trend filter ensures alignment with higher timeframe momentum. Volume confirms institutional participation.
# Target: 20-60 total trades over 4 years (5-15/year) to minimize fee drag.

name = "1d_Donchian_20_1wEMA50_Volume"
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
    
    # 1w data for Donchian and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper: highest high of last 20 weeks
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian lower: lowest low of last 20 weeks
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # 1-week EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Sufficient warmup for weekly indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above weekly Donchian upper, price above weekly EMA50, volume filter
            long_cond = (close[i] > donchian_upper_aligned[i]) and (close[i] > ema50_1w_aligned[i]) and volume_filter[i]
            # Short conditions: price breaks below weekly Donchian lower, price below weekly EMA50, volume filter
            short_cond = (close[i] < donchian_lower_aligned[i]) and (close[i] < ema50_1w_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below weekly Donchian lower
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above weekly Donchian upper
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals