#!/usr/bin/env python3
"""
1d_Vortex_Trend_Filter_WeeklyTrend
Hypothesis: Daily Vortex indicator (VI+ and VI-) captures trend direction and momentum. Combined with weekly trend filter (1w EMA50) and volume confirmation (volume > 2x 20-day average), this strategy identifies strong trend continuation moves. VI+ > VI- indicates bullish trend, VI- > VI+ indicates bearish trend. Works in both bull and bear markets by trading with the weekly trend direction while using Vortex for entry timing and strength confirmation.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Vortex Indicator (VI) on daily data
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    # Vortex movement
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus = np.concatenate([[np.nan], vm_plus[1:]])
    vm_minus = np.concatenate([[np.nan], vm_minus[1:]])
    
    # Smooth over 14 periods
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus14 / tr14
    vi_minus = vm_minus14 / tr14
    
    # Align weekly EMA to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: close > weekly EMA50 = bullish, < weekly EMA50 = bearish
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        # Vortex signals: VI+ > VI- = bullish momentum, VI- > VI+ = bearish momentum
        vortex_bullish = vi_plus[i] > vi_minus[i]
        vortex_bearish = vi_minus[i] > vi_plus[i]
        
        # Entry conditions with trend alignment and volume surge
        long_entry = (vortex_bullish and 
                     trend_up and 
                     volume_surge[i])
        
        short_entry = (vortex_bearish and 
                      trend_down and 
                      volume_surge[i])
        
        # Exit when Vortex reverses or trend changes
        long_exit = (vi_minus[i] > vi_plus[i]) or (close[i] < ema_50_1w_aligned[i])
        short_exit = (vi_plus[i] > vi_minus[i]) or (close[i] > ema_50_1w_aligned[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Vortex_Trend_Filter_WeeklyTrend"
timeframe = "1d"
leverage = 1.0