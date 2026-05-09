#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeFilter
# Hypothesis: Breakout above/below daily Camarilla R1/S1 levels with volume >1.5x 20-bar average and weekly trend filter.
# Weekly EMA34 ensures trend alignment: long only when weekly trend is up, short only when down.
# Designed for 15-25 trades/year on 1d timeframe to avoid fee drag and work in both bull and bear markets.

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(34) with proper initialization
    ema_34_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[0:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (close_1w[i] * 2 + ema_34_1w[i-1] * 32) / 34
    
    # Align weekly EMA to daily timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels from previous day's OHLC
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = np.nan
    high_1d_prev = np.roll(high_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev = np.roll(low_1d, 1)
    low_1d_prev[0] = np.nan
    
    camarilla_R1 = np.full_like(close_1d, np.nan)
    camarilla_S1 = np.full_like(close_1d, np.nan)
    
    valid = ~np.isnan(close_1d_prev) & ~np.isnan(high_1d_prev) & ~np.isnan(low_1d_prev)
    camarilla_R1[valid] = close_1d_prev[valid] + (high_1d_prev[valid] - low_1d_prev[valid]) * 1.1 / 12
    camarilla_S1[valid] = close_1d_prev[valid] - (high_1d_prev[valid] - low_1d_prev[valid]) * 1.1 / 12
    
    # Align Camarilla levels to daily timeframe (no shift needed as already daily)
    camarilla_R1_aligned = camarilla_R1  # Already on 1d timeframe
    camarilla_S1_aligned = camarilla_S1  # Already on 1d timeframe
    
    # Volume filter: daily volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or \
           np.isnan(camarilla_S1_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above R1 AND volume confirmation AND weekly bullish trend
            if close[i] > camarilla_R1_aligned[i] and volume_ratio[i] > 1.5 and ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below S1 AND volume confirmation AND weekly bearish trend
            elif close[i] < camarilla_S1_aligned[i] and volume_ratio[i] > 1.5 and ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below S1 (reversal signal) or weekly trend turns bearish
            if close[i] < camarilla_S1_aligned[i] or ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above R1 (reversal signal) or weekly trend turns bullish
            if close[i] > camarilla_R1_aligned[i] or ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals