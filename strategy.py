#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation.
Only trade in direction of 1d EMA200 (long when price > EMA200, short when price < EMA200).
Enter on Donchian breakout with volume > 1.5x 20-bar MA. Exit on opposite Donchian touch.
Designed for 4h timeframe to target 20-50 trades/year with strong trend capture and low fee drag.
Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate Donchian channels (20-period) on 4h data
    # We need at least 20 periods for Donchian, plus alignment buffer
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(200, 20)  # need EMA200 and Donchian lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_200_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND above 1d EMA200 AND volume spike
            if close[i] > highest_high[i] and close[i] > ema_200_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND below 1d EMA200 AND volume spike
            elif close[i] < lowest_low[i] and close[i] < ema_200_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price touches opposite Donchian level
            exit_signal = False
            if position == 1:
                # Exit long when price touches or goes below Donchian lower
                if close[i] <= lowest_low[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price touches or goes above Donchian upper
                if close[i] >= highest_high[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dEMA200_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0