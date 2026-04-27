#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeConfirm
Hypothesis: Uses daily Donchian(20) breakouts with weekly trend filter (price > weekly EMA50 for longs, price < weekly EMA50 for shorts) and volume confirmation (>1.5x 20-day average volume). Works in bull markets via breakouts above upper channel and in bear markets via breakdowns below lower channel. 1d timeframe targets 30-100 trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-day average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Donchian (20), weekly EMA50 (50), volume avg (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_max_val = high_max[i]
        low_min_val = low_min[i]
        ema_1w_val = ema_50_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long breakout: price above upper Donchian + weekly uptrend + volume
            if (close_val > high_max_val) and (close_val > ema_1w_val) and vol_conf:
                signals[i] = size
                position = 1
            # Short breakdown: price below lower Donchian + weekly downtrend + volume
            elif (close_val < low_min_val) and (close_val < ema_1w_val) and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly EMA50 or Donchian middle
            donchian_mid = (high_max_val + low_min_val) / 2
            exit_condition = (close_val < ema_1w_val) or (close_val < donchian_mid)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly EMA50 or Donchian middle
            donchian_mid = (high_max_val + low_min_val) / 2
            exit_condition = (close_val > ema_1w_val) or (close_val > donchian_mid)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0