#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDir_VolumeConfirm_v2
Hypothesis: 6h Donchian(20) breakout in direction of weekly Camarilla pivot bias (based on weekly close vs pivot) with 6h volume confirmation.
Weekly Camarilla pivot defines structural bias: price above weekly pivot = bullish bias, below = bearish bias.
Donchian breakouts aligned with weekly pivot bias have higher follow-through in both bull and bear markets.
Volume confirmation filters out low-conviction breakouts. Discrete sizing (0.25) limits fee drag.
Target: 50-150 total trades over 4 years (12-37/year) by requiring HTF alignment, breakout, and volume.
Uses actual weekly Camarilla calculation from weekly OHLC, not daily resampled.
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
    
    # Load weekly data ONCE before loop for HTF Camarilla pivot bias
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot and levels (based on previous week's OHLC)
    # We need to calculate for each weekly bar, then align to 6h
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Camarilla pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly Camarilla R1 and S1
    weekly_range = weekly_high - weekly_low
    camarilla_w_r1 = weekly_close + 1.1 * weekly_range / 12
    camarilla_w_s1 = weekly_close - 1.1 * weekly_range / 12
    
    # Weekly bias: 1 if close > pivot (bullish), -1 if close < pivot (bearish), 0 otherwise
    weekly_bias = np.where(weekly_close > weekly_pivot, 1, 
                          np.where(weekly_close < weekly_pivot, -1, 0))
    # Align weekly bias to 6h timeframe (completed weekly bars only)
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # 6h Donchian(20) - use rolling window on 6h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 6h volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian and volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_bias_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_above = close[i] > donchian_high[i-1]  # Break above previous period's high
        breakout_below = close[i] < donchian_low[i-1]   # Break below previous period's low
        
        if breakout_above and volume_spike:
            # Long signal: Donchian breakout above with volume, aligned with weekly bullish bias
            if weekly_bias_aligned[i] == 1:  # Weekly bias bullish
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            else:
                # Not aligned with weekly bias - hold or flatten
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = 0.0
                    position = 0
        elif breakout_below and volume_spike:
            # Short signal: Donchian breakout below with volume, aligned with weekly bearish bias
            if weekly_bias_aligned[i] == -1:  # Weekly bias bearish
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Not aligned with weekly bias - hold or flatten
                if position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
                    position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivotDir_VolumeConfirm_v2"
timeframe = "6h"
leverage = 1.0