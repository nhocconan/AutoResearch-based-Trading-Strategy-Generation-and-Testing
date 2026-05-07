#!/usr/bin/env python3
name = "6h_Fischer_Transform_WeeklyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly trend from daily close
    weekly_trend = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_trend_dir = np.where(weekly_trend > np.roll(weekly_trend, 5), 1, -1)  # 1=up, -1=down
    
    # Weekly trend aligned to 6h
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1d, weekly_trend_dir)
    
    # Fisher Transform on 6h prices (period=10)
    hl2 = (high + low) / 2
    max_hl2 = pd.Series(hl2).rolling(window=10, min_periods=10).max().values
    min_hl2 = pd.Series(hl2).rolling(window=10, min_periods=10).min().values
    range_hl2 = max_hl2 - min_hl2
    
    # Avoid division by zero
    value = np.where(range_hl2 != 0, 2 * ((hl2 - min_hl2) / range_hl2 - 0.5), 0)
    value = np.clip(value, -0.999, 0.999)
    
    fish = np.zeros_like(hl2)
    fish[0] = 0
    for i in range(1, n):
        fish[i] = 0.5 * np.log((1 + value[i]) / (1 - value[i])) + 0.5 * fish[i-1]
    
    # Volume filter: 2x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24
    
    for i in range(start_idx, n):
        if np.isnan(weekly_trend_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(fish[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Fisher crosses above -1.5 with weekly uptrend and volume
            if fish[i] > -1.5 and fish[i-1] <= -1.5 and weekly_trend_aligned[i] == 1 and volume[i] > vol_ma[i] * 2.0:
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses below 1.5 with weekly downtrend and volume
            elif fish[i] < 1.5 and fish[i-1] >= 1.5 and weekly_trend_aligned[i] == -1 and volume[i] > vol_ma[i] * 2.0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Fisher crosses below -1.0 or volume drops
            if fish[i] < -1.0 or volume[i] < vol_ma[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Fisher crosses above 1.0 or volume drops
            if fish[i] > 1.0 or volume[i] < vol_ma[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Fischer Transform reversals + weekly trend filter + volume confirmation
# Fischer Transform identifies extreme price movements likely to reverse.
# Weekly trend (50-period EMA direction) ensures trades align with higher timeframe momentum.
# Volume spike (2x average) confirms institutional participation.
# Works in bull (buy Fischer > -1.5 in uptrend) and bear (sell Fischer < 1.5 in downtrend).
# Conservative position sizing (0.25) targets 15-30 trades/year to minimize fee drag.