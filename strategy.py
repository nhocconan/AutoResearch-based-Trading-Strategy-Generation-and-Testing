#!/usr/bin/env python3
"""
12h_Donchian20_1dTrend_VolumeSpike
Hypothesis: Donchian(20) breakout on 12h with 1d EMA34 trend filter and volume spike confirmation.
Trades with the trend in both bull and bear markets using 1d EMA34 as trend filter.
Targets 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian channels (20-period)
        donchian_high = np.max(high[i-20:i])
        donchian_low = np.min(low[i-20:i])
        
        # Trend direction from 1d EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: >1.8x 20-period MA
        vol_confirm = volume[i] > (1.8 * vol_ma_20[i])
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high
        short_breakout = close[i] < donchian_low
        
        # Entry logic
        long_entry = vol_confirm and trend_up and long_breakout
        short_entry = vol_confirm and trend_down and short_breakout
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = (close[i] < donchian_low) or (not trend_up)
        short_exit = (close[i] > donchian_high) or (not trend_down)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0