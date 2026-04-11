#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly trend direction from 1W close above/below SMA50
# and daily mean reversion at Bollinger Bands (20,2) with volume confirmation.
# Long when: weekly trend up, price touches lower BB, volume > 1.5x 20-period average.
# Short when: weekly trend down, price touches upper BB, volume > 1.5x 20-period average.
# Exit when price crosses 20-period SMA or opposite BB touch.
# Designed for 12-37 trades/year with controlled risk in both bull and bear markets.

name = "6h_1w1d_bb_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly trend: close above/below 50-period SMA
    close_1w = df_1w['close'].values
    sma50_1w = np.full_like(close_1w, np.nan)
    for i in range(49, len(close_1w)):
        sma50_1w[i] = np.mean(close_1w[i-49:i+1])
    
    weekly_trend_up = close_1w > sma50_1w
    weekly_trend_down = close_1w < sma50_1w
    
    # Align weekly trend to 6h
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Daily Bollinger Bands (20,2)
    close_1d = df_1d['close'].values
    sma20_1d = np.full_like(close_1d, np.nan)
    std20_1d = np.full_like(close_1d, np.nan)
    for i in range(19, len(close_1d)):
        sma20_1d[i] = np.mean(close_1d[i-19:i+1])
        std20_1d[i] = np.std(close_1d[i-19:i+1])
    
    upper_bb = sma20_1d + 2 * std20_1d
    lower_bb = sma20_1d - 2 * std20_1d
    
    # Daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align daily indicators to 6h
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma20_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(sma20_1d_aligned[i]) or np.isnan(vol_avg_aligned[i]) or
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Determine weekly trend direction
        is_up_trend = weekly_trend_up_aligned[i]
        is_down_trend = weekly_trend_down_aligned[i]
        
        # Mean reversion signals
        bb_lower_touch = low[i] <= lower_bb_aligned[i]
        bb_upper_touch = high[i] >= upper_bb_aligned[i]
        
        # Exit conditions
        cross_sma = (position == 1 and close[i] <= sma20_1d_aligned[i]) or \
                    (position == -1 and close[i] >= sma20_1d_aligned[i])
        opposite_bb = (position == 1 and high[i] >= upper_bb_aligned[i]) or \
                      (position == -1 and low[i] <= lower_bb_aligned[i])
        
        # Entry logic
        if is_up_trend and bb_lower_touch and vol_filter and position != 1:
            position = 1
            signals[i] = 0.25
        elif is_down_trend and bb_upper_touch and vol_filter and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (cross_sma or opposite_bb):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (cross_sma or opposite_bb):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals