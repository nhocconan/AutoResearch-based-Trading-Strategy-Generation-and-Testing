#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: Trade daily Donchian(20) breakouts with 1-week EMA50 trend filter and volume confirmation (>2.0x 20-bar MA).
Daily timeframe targets 7-25 trades/year to minimize fee drag. Donchian channels provide strong breakout levels.
1-week EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws.
Volume confirmation adds conviction to breakouts. Discrete sizing 0.25 balances profit and fee drag.
Works in bull/bear: trend filter adapts to market direction, volume confirms breakout validity.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on daily data
    # We need to calculate on 1d data then align, but since we're on 1d timeframe,
    # we can calculate directly on the prices data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll  # Upper band
    donchian_low = low_roll    # Lower band
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20) and volume MA (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND 1w trend bullish (close > EMA50) AND volume confirm
            long_setup = (close[i] > donchian_high[i]) and \
                         (close[i] > ema_50_1w_aligned[i]) and \
                         volume_confirm[i]
            # Short: price breaks below Donchian lower band AND 1w trend bearish (close < EMA50) AND volume confirm
            short_setup = (close[i] < donchian_low[i]) and \
                          (close[i] < ema_50_1w_aligned[i]) and \
                          volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Donchian channel OR 1w trend turns bearish
            if (close[i] < donchian_high[i] and close[i] > donchian_low[i]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel OR 1w trend turns bullish
            if (close[i] < donchian_high[i] and close[i] > donchian_low[i]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0