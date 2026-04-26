#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_Filter_VolumeSpike
Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation (>1.5x average volume).
In bull markets: price breaks above 20-day high with weekly uptrend and high volume → long.
In bear markets: price breaks below 20-day low with weekly downtrend and high volume → short.
Uses discrete position sizing (0.30) to balance return and risk. Target: 30-100 trades over 4 years (7-25/year) on 1d timeframe.
Weekly trend filter ensures we only trade with the dominant higher-timeframe momentum, reducing whipsaw.
Volume spike confirms institutional participation. Works on BTC/ETH by requiring weekly trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for Donchian and volume
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need minimum for trend
        return np.zeros(n)
    
    # Weekly EMA10 for trend filter (responsive but smoothed)
    ema_10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Start after warmup (need 20 for Donchian, 20 for volume)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Donchian(20) levels: highest high and lowest low of past 20 days (excluding current)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_10_1w_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(highest_high) or np.isnan(lowest_low) or 
            np.isnan(ema_val) or np.isnan(avg_vol)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume (balanced for trade frequency)
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price breaks above 20-day high with weekly uptrend and volume confirmation
        long_condition = (close_val > highest_high) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below 20-day low with weekly downtrend and volume confirmation
        short_condition = (close_val < lowest_low) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: opposite Donchian breakout or trend reversal
        exit_long = (close_val < lowest_low) or (close_val < ema_val)
        exit_short = (close_val > highest_high) or (close_val > ema_val)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_Filter_VolumeSpike"
timeframe = "1d"
leverage = 1.0