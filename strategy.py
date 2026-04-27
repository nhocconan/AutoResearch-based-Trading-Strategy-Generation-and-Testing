#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_Volume
Hypothesis: Uses 12-hour Donchian(20) breakouts confirmed by daily EMA trend and volume spikes.
Designed to capture major trend moves with low trade frequency (~15-25 trades/year) to minimize fee drag.
Works in both bull and bear markets by following the daily trend direction, avoiding counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Daily EMA trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = max(lookback, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        ema_trend = ema_34_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above upper channel with uptrend and volume
            if close_val > upper_channel and close_val > ema_trend and vol_conf:
                signals[i] = size
                position = 1
            # Short: price breaks below lower channel with downtrend and volume
            elif close_val < lower_channel and close_val < ema_trend and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower channel (reversal signal)
            if close_val < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above upper channel (reversal signal)
            if close_val > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0