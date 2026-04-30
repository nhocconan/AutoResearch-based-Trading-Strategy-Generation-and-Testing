#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Uses weekly Camarilla pivots for major trend structure (avoids counter-trend in strong weekly trends),
# 6h Donchian breakout for entry timing, and volume > 1.8x 20-bar average for confirmation.
# Discrete position sizing at ±0.25 to limit fee drag. Target: 80-140 total trades over 4 years (20-35/year).
# Session filter (08:00-20:00 UTC) to avoid low-liquidity periods.

name = "6h_Donchian20_WeeklyCamarilla_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load weekly data ONCE before loop for Camarilla pivots (major trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from prior 1w OHLC (use shift(1) to avoid look-ahead)
    high_1w = df_1w['high'].shift(1).values
    low_1w = df_1w['low'].shift(1).values
    close_1w = df_1w['close'].shift(1).values
    
    # Weekly Camarilla R3, S3 (primary breakout levels)
    camarilla_range = high_1w - low_1w
    camarilla_r3 = close_1w + (1.1 * camarilla_range * 1.1 / 4)
    camarilla_s3 = close_1w - (1.1 * camarilla_range * 1.1 / 4)
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Donchian(20) channels for breakout detection
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian(20) high, above weekly R3, volume confirmation
            if (curr_high > highest_high[i] and 
                curr_close > curr_r3 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian(20) low, below weekly S3, volume confirmation
            elif (curr_low < lowest_low[i] and 
                  curr_close < curr_s3 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian(20) low OR weekly S3
            if (curr_low < lowest_low[i] or 
                curr_close < curr_s3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian(20) high OR weekly R3
            if (curr_high > highest_high[i] or 
                curr_close > curr_r3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals