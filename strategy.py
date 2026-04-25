#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_TrendFilter_VolumeSpike
Hypothesis: Trade 1d timeframe using weekly Donchian(20) breakouts filtered by 1w EMA34 trend and volume spikes.
Enter long when price breaks above weekly Donchian high AND 1w trend is bullish (close > EMA34) AND volume > 1.5x 20-day average.
Enter short when price breaks below weekly Donchian low AND 1w trend is bearish (close < EMA34) AND volume > 1.5x 20-day average.
Exit when price re-enters the weekly Donchian channel or 1w trend reverses.
Uses discrete sizing 0.25 to manage risk and minimize fee churn. Target 20-50 trades/year on 1d timeframe.
Weekly Donchian provides structure on higher timeframe, reducing noise. 1w EMA34 filter ensures we only trade with the higher timeframe trend.
Volume spike confirmation adds conviction to breakouts, filtering out false signals. Designed to work in both bull and bear markets by following the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly Donchian(20) channels
    # Donchian high = max(high, lookback=20)
    # Donchian low = min(low, lookback=20)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Volume spike: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly EMA34 (34) and Donchian (20) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high AND weekly trend bullish (close > EMA34) AND volume spike
            long_setup = (close[i] > donchian_high_aligned[i]) and \
                         (close[i] > ema_34_1w_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below weekly Donchian low AND weekly trend bearish (close < EMA34) AND volume spike
            short_setup = (close[i] < donchian_low_aligned[i]) and \
                          (close[i] < ema_34_1w_aligned[i]) and \
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
            # Exit: price re-enters weekly Donchian channel (closes below Donchian high AND above Donchian low) OR weekly trend turns bearish
            if (close[i] < donchian_high_aligned[i] and close[i] > donchian_low_aligned[i]) or \
               (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters weekly Donchian channel (closes below Donchian high AND above Donchian low) OR weekly trend turns bullish
            if (close[i] < donchian_high_aligned[i] and close[i] > donchian_low_aligned[i]) or \
               (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_TrendFilter_VolumeSpike"
timeframe = "1d"
leverage = 1.0