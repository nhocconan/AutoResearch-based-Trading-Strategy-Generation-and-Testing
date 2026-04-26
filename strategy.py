#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: 12h Donchian(20) breakout in direction of 1d EMA50 trend with volume confirmation (>1.5x average) captures strong momentum moves while avoiding false breakouts. Works in bull/bear via 1d trend filter. Designed for 12h to target 12-37 trades/year with discrete sizing (0.25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) channels from 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Average volume for confirmation (24-period SMA = 2d)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Warmup: max of Donchian(20), EMA(50), volume(24)
    start_idx = max(20, 50, 24)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1d_aligned[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(upper) or 
            np.isnan(lower)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Trend filter: price vs 1d EMA50
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Long: price CLOSES above upper Donchian with 1d uptrend and volume
        long_condition = (close_val > upper) and uptrend and volume_confirmed
        # Short: price CLOSES below lower Donchian with 1d downtrend and volume
        short_condition = (close_val < lower) and downtrend and volume_confirmed
        
        # Exit: price retests middle of channel (mean reversion)
        mid_channel = (upper + lower) / 2
        long_exit = (position == 1 and close_val < mid_channel)
        short_exit = (position == -1 and close_val > mid_channel)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0