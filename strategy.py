#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_1wPivotDirection_VolumeConfirm
Hypothesis: Trade 6h Donchian(20) breakouts with 1d EMA34 trend filter and 1w pivot direction confirmation.
Donchian breakouts capture momentum; 1d EMA34 ensures trading with daily trend; 1w pivot (PP) provides
higher timeframe bias (long only above weekly PP, short only below). Volume confirmation (>1.5x 20-bar MA)
adds conviction. 6h timeframe targets 12-37 trades/year to minimize fee drag. Works in bull/bear: 
trend filter adapts to market direction, weekly pivot filters counter-trend breakouts, volume confirms validity.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Get 1w data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly pivot point (PP) = (H+L+C)/3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # Calculate Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA34 (34), Donchian (20), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(pp_1w_aligned[i]) or
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND 1d trend bullish AND price > weekly PP AND volume confirm
            long_setup = (close[i] > highest_high_20[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         (close[i] > pp_1w_aligned[i]) and \
                         volume_confirm[i]
            # Short: price breaks below Donchian lower AND 1d trend bearish AND price < weekly PP AND volume confirm
            short_setup = (close[i] < lowest_low_20[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          (close[i] < pp_1w_aligned[i]) and \
                          volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Donchian channel OR 1d trend turns bearish OR price < weekly PP
            if (close[i] < highest_high_20[i] and close[i] > lowest_low_20[i]) or \
               (close[i] < ema_34_1d_aligned[i]) or \
               (close[i] < pp_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel OR 1d trend turns bullish OR price > weekly PP
            if (close[i] < highest_high_20[i] and close[i] > lowest_low_20[i]) or \
               (close[i] > ema_34_1d_aligned[i]) or \
               (close[i] > pp_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_1wPivotDirection_VolumeConfirm"
timeframe = "6h"
leverage = 1.0