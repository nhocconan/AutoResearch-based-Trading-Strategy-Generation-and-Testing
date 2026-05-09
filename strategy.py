#!/usr/bin/env python3
# 4h_RangeReversal_Bollinger_Bands_2std
# Hypothesis: Mean reversion at Bollinger Bands (20,2) with volume confirmation and trend filter from 1d EMA50.
# In ranging markets, price reverts from bands; in trends, only trade with trend (price > EMA50 for long, < EMA50 for short).
# Uses Bollinger Bands for entry/exit, volume > 1.3x 20-bar average for confirmation, and 1d EMA50 for trend filter.
# Designed for 20-40 trades/year on 4h timeframe to avoid fee drag.

name = "4h_RangeReversal_Bollinger_Bands_2std"
timeframe = "4h"
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
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    bb_ma = np.full_like(close, np.nan)
    bb_stddev = np.full_like(close, np.nan)
    if len(close) >= bb_period:
        for i in range(bb_period - 1, len(close)):
            bb_ma[i] = np.mean(close[i - bb_period + 1:i + 1])
            bb_stddev[i] = np.std(close[i - bb_period + 1:i + 1])
    bb_upper = bb_ma + bb_std * bb_stddev
    bb_lower = bb_ma - bb_std * bb_stddev
    
    # Volume filter: 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma[i] = np.mean(volume[i - 19:i + 1])
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA(50)
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(volume_ratio[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price touches/below lower BB AND volume confirmation AND bullish trend (price > EMA50)
            if close[i] <= bb_lower[i] and volume_ratio[i] > 1.3 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price touches/above upper BB AND volume confirmation AND bearish trend (price < EMA50)
            elif close[i] >= bb_upper[i] and volume_ratio[i] > 1.3 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses above the middle band (mean reversion complete) or trend turns bearish
            if close[i] >= bb_ma[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses below the middle band or trend turns bullish
            if close[i] <= bb_ma[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals