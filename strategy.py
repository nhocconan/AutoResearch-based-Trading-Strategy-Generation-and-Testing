#!/usr/bin/env python3
# 6h_WeeklyTrend_DailyPullback_Entry
# Hypothesis: Trend-following strategy that uses weekly trend direction (EMA34) and enters on daily pullbacks to EMA21.
# Works in both bull and bear markets by only taking trades in the direction of the weekly trend.
# Uses 6h timeframe for entry timing with volume confirmation to reduce false signals.
# Target: 20-35 trades/year per symbol with disciplined risk management.

name = "6h_WeeklyTrend_DailyPullback_Entry"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for trend direction (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34
    ema_34_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[0:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (close_1w[i] * 2 + ema_34_1w[i-1] * 33) / 35
    
    # Get daily data for pullback entry (EMA21) and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA21
    ema_21_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 21:
        ema_21_1d[20] = np.mean(close_1d[0:21])
        for i in range(21, len(close_1d)):
            ema_21_1d[i] = (close_1d[i] * 2 + ema_21_1d[i-1] * 20) / 22
    
    # Calculate daily average volume (20-period)
    avg_volume_20d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        avg_volume_20d[19] = np.mean(volume_1d[0:20])
        for i in range(20, len(volume_1d)):
            avg_volume_20d[i] = (volume_1d[i] * 19 + avg_volume_20d[i-1]) / 20
    
    # Align weekly trend to 6h timeframe
    weekly_trend_up = ema_34_1w > np.roll(ema_34_1w, 1)  # Rising EMA
    weekly_trend_up[0] = False  # First value has no previous
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    
    # Align daily EMA21 to 6h timeframe
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Align daily average volume to 6h timeframe
    avg_volume_20d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 22, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(weekly_trend_up_aligned[i]) or np.isnan(ema_21_1d_aligned[i]) or \
           np.isnan(avg_volume_20d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current 6h volume > 1.5x daily average volume (scaled)
        # Scale daily volume to 6h approximation: daily volume / 4 (since 4x6h in a day)
        volume_condition = volume[i] > (avg_volume_20d_aligned[i] / 4.0) * 1.5
        
        if position == 0:
            # Enter long: Weekly trend UP AND price pulls back to/touches daily EMA21 AND volume confirmation
            if weekly_trend_up_aligned[i] and close[i] <= ema_21_1d_aligned[i] * 1.001 and volume_condition:
                signals[i] = 0.25
                position = 1
            # Enter short: Weekly trend DOWN AND price pulls back to/touches daily EMA21 AND volume confirmation
            elif not weekly_trend_up_aligned[i] and close[i] >= ema_21_1d_aligned[i] * 0.999 and volume_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Weekly trend turns down OR price breaks above daily EMA21 by 0.5%
            if not weekly_trend_up_aligned[i] or close[i] > ema_21_1d_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Weekly trend turns up OR price breaks below daily EMA21 by 0.5%
            if weekly_trend_up_aligned[i] or close[i] < ema_21_1d_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals