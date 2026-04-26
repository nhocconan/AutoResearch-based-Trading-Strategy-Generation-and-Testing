#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_v1
Hypothesis: Trade 6h Donchian(20) breakouts filtered by weekly pivot levels (PP, R1, S1) and 1w EMA(50) trend.
Long when price breaks above Donchian(20) high AND close > weekly PP AND 1w EMA(50) up.
Short when breaks below Donchian(20) low AND close < weekly PP AND 1w EMA(50) down.
Volume confirmation required. Targets 50-150 trades over 4 years. Works in bull/bear via 1w trend filter and pivot bias.
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
    
    # Get 1w data for trend filter and weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly pivot from previous 1w bar
    # PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    prev_close_1w = df_1w['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high_1w = np.where(np.isnan(prev_high_1w), df_1w['high'].values, prev_high_1w)
    prev_low_1w = np.where(np.isnan(prev_low_1w), df_1w['low'].values, prev_low_1w)
    prev_close_1w = np.where(np.isnan(prev_close_1w), df_1w['close'].values, prev_close_1w)
    
    pp_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    r1_1w = 2 * pp_1w - prev_low_1w
    s1_1w = 2 * pp_1w - prev_high_1w
    
    # Align weekly pivot levels to 6h
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Donchian(20) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1w EMA(50), Donchian(20), volume MA(20)
    start_idx = max(50, lookback, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(pp_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        trend_up = close_val > ema_50_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_50_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: price breaks above Donchian high AND close > weekly PP AND 1w uptrend AND volume
            long_signal = (close_val > highest_high[i]) and (close_val > pp_1w_aligned[i]) and trend_up and vol_conf
            
            # Short: price breaks below Donchian low AND close < weekly PP AND 1w downtrend AND volume
            short_signal = (close_val < lowest_low[i]) and (close_val < pp_1w_aligned[i]) and trend_down and vol_conf
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price drops below Donchian low (failed breakout) OR 1w trend flips down OR close < weekly PP
            if (close_val < lowest_low[i]) or (not trend_up) or (close_val < pp_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Donchian high (failed breakdown) OR 1w trend flips up OR close > weekly PP
            if (close_val > highest_high[i]) or (not trend_down) or (close_val > pp_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout_v1"
timeframe = "6h"
leverage = 1.0