#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyTrend_VolumeSpike
Hypothesis: Trade daily Donchian(20) breakouts in direction of weekly trend with volume confirmation.
Weekly EMA50 trend filter avoids counter-trend trades in bear markets. Donchian breakouts capture momentum.
Volume spike (2x 20-day average) confirms institutional interest. Discrete sizing 0.25 minimizes fee churn.
Target: 15-25 trades/year to stay within fee drag limits for 1d timeframe.
Works in both bull (trend continuation) and bear (trend reversals on weekly timeframe).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    # Use rolling window on daily high/low - but we need to get daily data first
    # Since we're on 1d timeframe, we can use prices directly
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume spike: current volume > 2.0x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly EMA50 (50) and Donchian (20) and volume MA (20)
    start_idx = max(50, donchian_window, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND weekly trend bullish (close > EMA50) AND volume spike
            long_setup = (high[i] > donchian_high[i]) and \
                         (close[i] > ema_50_1w_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below Donchian low AND weekly trend bearish (close < EMA50) AND volume spike
            short_setup = (low[i] < donchian_low[i]) and \
                          (close[i] < ema_50_1w_aligned[i]) and \
                          volume_spike[i]
            
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
            # Exit: price re-enters Donchian channel OR weekly trend turns bearish
            if (low[i] < donchian_high[i] and high[i] > donchian_low[i]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel OR weekly trend turns bullish
            if (low[i] < donchian_high[i] and high[i] > donchian_low[i]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0