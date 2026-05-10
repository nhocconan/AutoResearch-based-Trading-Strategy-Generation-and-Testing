#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_VolumeConfirm
# Hypothesis: Uses weekly pivot points (S1, R1, S2, R2) as key support/resistance levels.
# Long when price breaks above weekly R1 with volume > 1.5x average and price > 1d EMA50.
# Short when price breaks below weekly S1 with volume > 1.5x average and price < 1d EMA50.
# Uses 6h Donchian(20) breakout direction to avoid false breaks in ranging markets.
# Designed for 15-25 trades/year to avoid overtrading and work in both bull and bear markets.

name = "6h_WeeklyPivot_DonchianBreakout_VolumeConfirm"
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
    
    # Calculate weekly OHLC for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly OHLC for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot Points: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H-L), S2 = P - (H-L)
    pivot = (high_1w + low_1w + close_1w) / 3
    weekly_r1 = 2 * pivot - low_1w
    weekly_s1 = 2 * pivot - high_1w
    weekly_r2 = pivot + (high_1w - low_1w)
    weekly_s2 = pivot - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Donchian(20) for breakout direction
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume confirmation and Donchian breakout
            if (close[i] > weekly_r1_aligned[i] and
                volume[i] > 1.5 * vol_ma[i] and
                close[i] > ema_50_1d_aligned[i] and
                close[i] > donchian_high[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume confirmation and Donchian breakout
            elif (close[i] < weekly_s1_aligned[i] and
                  volume[i] > 1.5 * vol_ma[i] and
                  close[i] < ema_50_1d_aligned[i] and
                  close[i] < donchian_low[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses below weekly S2 or Donchian low
            if close[i] < weekly_s2_aligned[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above weekly R2 or Donchian high
            if close[i] > weekly_r2_aligned[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals