#!/usr/bin/env python3
"""
4h_Vortex_Trend_Filter_Volume_Spike
Hypothesis: Vortex indicator (VI+) crossing above VI- signals trend start, confirmed by volume spike and price above/below 4h EMA34. Works in both bull and bear by capturing new trends early. Targets 20-40 trades/year via strict entry conditions.
"""

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
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Vortex Indicator (14-period)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    vm_plus = np.abs(high - low[1:])
    vm_plus = np.concatenate([[np.nan], vm_plus[:-1]])
    vm_minus = np.abs(low - high[1:])
    vm_minus = np.concatenate([[np.nan], vm_minus[:-1]])
    
    # Sum over 14 periods
    n14 = 14
    tr_sum = pd.Series(tr).rolling(window=n14, min_periods=n14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=n14, min_periods=n14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=n14, min_periods=n14).sum().values
    
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]):
            signals[i] = 0.0
            continue
        
        # Trend direction from 4h EMA34
        trend_up = close[i] > ema_34_4h_aligned[i]
        trend_down = close[i] < ema_34_4h_aligned[i]
        
        # Volume confirmation: >2.0x 20-period MA
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Vortex crossover signals
        vi_cross_up = vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1]
        vi_cross_down = vi_plus[i] < vi_minus[i] and vi_plus[i-1] >= vi_minus[i-1]
        
        # Entry logic: Vortex crossover + volume + trend alignment
        long_entry = vi_cross_up and vol_confirm and trend_up
        short_entry = vi_cross_down and vol_confirm and trend_down
        
        # Exit logic: Vortex reverse crossover or trend violation
        long_exit = vi_cross_down or (not trend_up)
        short_exit = vi_cross_up or (not trend_down)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Vortex_Trend_Filter_Volume_Spike"
timeframe = "4h"
leverage = 1.0