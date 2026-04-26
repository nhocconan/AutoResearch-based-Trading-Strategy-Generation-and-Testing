#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: 4h Camarilla R1/S1 breakout in direction of 1d EMA34 trend with volume confirmation.
Combines proven winning pattern: price channel breakout + HTF trend filter + volume spike.
Uses discrete sizing (0.25) to limit fee drag. Target: 75-200 total trades over 4 years.
Works in bull/bear: EMA34 trend filter adapts to regime, volume confirmation ensures conviction.
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
    
    # Load 1d data ONCE before loop for HTF EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 trend
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d trend: 1 if close > EMA34 (bullish), -1 if close < EMA34 (bearish)
    trend_1d = np.where(close_1d > ema_34_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate daily Camarilla pivot levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot = (high + low + close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Daily Camarilla R1 and S1
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * range_1d / 12
    camarilla_s1 = close_1d - 1.1 * range_1d / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 4h Donchian-like breakout: use Camarilla levels as breakout thresholds
    # We'll use the Camarilla levels directly for breakout signals
    
    # 4h volume confirmation: volume > 1.8x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trend_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (stricter: 1.8x average)
        volume_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        # Breakout conditions using Camarilla levels
        breakout_above = close[i] > camarilla_r1_aligned[i]
        breakout_below = close[i] < camarilla_s1_aligned[i]
        
        if breakout_above and volume_spike:
            # Long signal: Camarilla R1 breakout with volume, aligned with 1d bullish trend
            if trend_1d_aligned[i] == 1:  # 1d trend bullish
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            else:
                # Not aligned with 1d trend - hold or flatten
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = 0.0
                    position = 0
        elif breakout_below and volume_spike:
            # Short signal: Camarilla S1 breakout with volume, aligned with 1d bearish trend
            if trend_1d_aligned[i] == -1:  # 1d trend bearish
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Not aligned with 1d trend - hold or flatten
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

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0