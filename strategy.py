#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_WeeklyTrend_VolumeSpike_v1
Hypothesis: Daily Donchian(20) breakout in direction of weekly EMA50 trend with volume confirmation.
Weekly Donchian channels provide robust structural support/resistance less prone to whipsaw than intraday levels.
Weekly EMA50 trend filter ensures alignment with higher timeframe momentum, reducing counter-trend trades.
Volume confirmation adds conviction filter. Discrete sizing (0.25) limits fee drag.
Target: 30-100 total trades over 4 years (7-25/year) by requiring weekly alignment, Donchian breakout, trend alignment, and volume.
Works in both bull and bear markets by only trading with the weekly trend and requiring volume confirmation to avoid false breakouts.
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
    
    # Load weekly data ONCE before loop for HTF Donchian and EMA
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period high/low)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly Donchian upper and lower bands (20-period)
    dh_20 = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    dl_20 = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA50 for trend filter
    weekly_ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to daily timeframe (completed weekly bars only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, dh_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, dl_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_50)
    
    # Daily volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian and volume MA, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (strict to reduce trades)
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_above = close[i] > donchian_high_aligned[i]  # Break above weekly Donchian high
        breakout_below = close[i] < donchian_low_aligned[i]   # Break below weekly Donchian low
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if breakout_above and volume_spike and uptrend:
            # Long signal: Weekly Donchian breakout with volume, in weekly uptrend
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif breakout_below and volume_spike and downtrend:
            # Short signal: Weekly Donchian breakout with volume, in weekly downtrend
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_WeeklyTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0