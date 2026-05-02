#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot filter and volume confirmation
# Uses 6h primary timeframe to balance signal frequency and noise reduction
# Weekly pivot levels provide institutional reference points from higher timeframe
# Donchian breakout captures momentum with weekly pivot as trend filter (only trade in direction of weekly bias)
# Volume confirmation (>1.5 * 20-period EMA) ensures institutional participation
# Designed for low trade frequency: ~10-20 trades/year per symbol with 0.25 sizing
# Weekly pivot acts as regime filter: only long above weekly pivot, short below
# Works in bull markets via breakout continuation and bear markets via breakdown continuation
# Avoids overtrading by requiring confluence of price level, weekly bias, and volume

name = "6h_Donchian20_WeeklyPivot_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly HTF data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader pivots)
    # Based on previous weekly bar
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close[0] = np.nan
    prev_weekly_high[0] = np.nan
    prev_weekly_low[0] = np.nan
    
    # Weekly pivot calculations
    weekly_pivot = np.full(len(weekly_close), np.nan)
    weekly_r1 = np.full(len(weekly_close), np.nan)
    weekly_s1 = np.full(len(weekly_close), np.nan)
    weekly_r2 = np.full(len(weekly_close), np.nan)
    weekly_s2 = np.full(len(weekly_close), np.nan)
    
    for i in range(1, len(weekly_close)):
        weekly_pivot[i] = (prev_weekly_high[i] + prev_weekly_low[i] + prev_weekly_close[i]) / 3.0
        weekly_r1[i] = 2 * weekly_pivot[i] - prev_weekly_low[i]
        weekly_s1[i] = 2 * weekly_pivot[i] - prev_weekly_high[i]
        weekly_r2[i] = weekly_pivot[i] + (prev_weekly_high[i] - prev_weekly_low[i])
        weekly_s2[i] = weekly_pivot[i] - (prev_weekly_high[i] - prev_weekly_low[i])
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Donchian(20) channels on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = max(50, lookback)
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND above weekly pivot AND volume spike
            if (high[i] > highest_high[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND below weekly pivot AND volume spike
            elif (low[i] < lowest_low[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR price below weekly pivot
            if low[i] < lowest_low[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR price above weekly pivot
            if high[i] > highest_high[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals