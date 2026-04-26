#!/usr/bin/env python3
"""
6h_WeeklyDonchian_Breakout_DailyTrend_VolumeFilter
Hypothesis: 6h breakout of weekly Donchian(20) channels with daily EMA50 trend filter and volume confirmation.
Long when price breaks above weekly Donchian high in daily uptrend with volume > 1.5x 20-period MA.
Short when price breaks below weekly Donchian low in daily downtrend with volume spike.
Uses weekly structure for fewer, higher-quality signals (target: 12-30 trades/year).
Daily trend filter ensures alignment with intermediate-term momentum.
Volume confirmation reduces false breakouts.
Works in both bull and bear markets by following the daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need warmup for weekly Donchian and volume MA
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels (prior completed weekly candle)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need 20 periods for Donchian
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need 50 for EMA
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high: rolling max of 20 weekly highs
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of 20 weekly lows
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # Daily EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close > ema_50_1d_aligned
    downtrend_1d = close < ema_50_1d_aligned
    
    # Volume confirmation: volume > 1.5x 20-period MA (moderate threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for weekly Donchian and volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with daily uptrend and volume spike
            if (close[i] > donchian_high_aligned[i] and 
                uptrend_1d[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with daily downtrend and volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  downtrend_1d[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below weekly Donchian low OR daily trend changes to downtrend
            if (close[i] < donchian_low_aligned[i] or not uptrend_1d[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above weekly Donchian high OR daily trend changes to uptrend
            if (close[i] > donchian_high_aligned[i] or not downtrend_1d[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyDonchian_Breakout_DailyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0