#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot filter and volume confirmation
# Long when price breaks above Donchian(20) high AND price > weekly pivot (PP) AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low AND price < weekly pivot (PP) AND volume > 1.5x 20-period average
# Exit when price crosses back to opposite Donchian level OR weekly pivot filter flips
# Uses 6h primary timeframe with 1d HTF for weekly pivot filter to reduce whipsaw and capture institutional levels
# Discrete sizing (0.25) to limit fee drag and manage drawdown in ranging markets (2025+ test)
# Novelty: Weekly pivot (not camarilla) as HTF filter on 6h Donchian breakout - untried combination per session history

name = "6h_Donchian20_1dWeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 days for meaningful weekly pivot
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's 1d data
    # Weekly pivot: PP = (Weekly High + Weekly Low + Weekly Close) / 3
    # We approximate weekly by rolling 5-day window on 1d data (trading week)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    if len(high_1d) >= 5:
        # Rolling window of 5 days for weekly aggregation
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot point from prior week (shifted by 1 to avoid look-ahead)
        weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_pp_prior = np.roll(weekly_pp, 1)
        weekly_pp_prior[0] = np.nan
    else:
        weekly_pp_prior = np.full(len(high_1d), np.nan)
    
    # Align weekly pivot to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp_prior)
    
    # Donchian(20) channels on 6h data
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average (balanced for 6f)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pp_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND price > weekly PP AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > weekly_pp_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND price < weekly PP AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < weekly_pp_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian low OR price < weekly PP (filter flip)
            if (close[i] < donchian_low[i] or 
                close[i] < weekly_pp_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian high OR price > weekly PP (filter flip)
            if (close[i] > donchian_high[i] or 
                close[i] > weekly_pp_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals