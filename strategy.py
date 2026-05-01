#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Donchian breakouts capture strong momentum moves. Weekly pivot (from prior week) provides
# institutional reference: break above weekly R1 = bullish bias, break below weekly S1 = bearish bias.
# Volume confirmation ensures breakouts have conviction. Works in bull (breakouts with volume)
# and bear (trend continuation after pullbacks). Discrete sizing (0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeSpike_v1"
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
    
    # 1w HTF data for weekly pivot calculation (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot levels from prior 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot formulas: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_r1 = 2 * weekly_pivot - low_1w   # R1
    weekly_s1 = 2 * weekly_pivot - high_1w  # S1
    
    # Align weekly pivot levels to 6h timeframe (using prior week's levels)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Donchian(20) channels on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(lookback, 20)  # Need 20 for Donchian + 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Donchian breakout conditions (using prior bar levels to avoid look-ahead)
        breakout_up = curr_close > highest_high[i-1]  # Break above 20-period high
        breakout_down = curr_close < lowest_low[i-1]  # Break below 20-period low
        
        # Volume confirmation and weekly pivot direction filter
        vol_spike = volume_spike[i]
        # Weekly pivot filter: price above/below weekly R1/S1 for directional bias
        bullish_bias = curr_close > weekly_r1_aligned[i]
        bearish_bias = curr_close < weekly_s1_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up, volume spike, bullish bias from weekly pivot
            if breakout_up and vol_spike and bullish_bias:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down, volume spike, bearish bias from weekly pivot
            elif breakout_down and vol_spike and bearish_bias:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown or loss of bullish bias
            if curr_close < lowest_low[i] or curr_close < weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout or loss of bearish bias
            if curr_close > highest_high[i] or curr_close > weekly_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals