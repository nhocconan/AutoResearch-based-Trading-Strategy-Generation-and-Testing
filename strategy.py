#!/usr/bin/env python3
# 1d_Touchstone_Signal
# Hypothesis: Combines 1-week Donchian breakout with 1-day Bollinger mean-reversion within the weekly range.
# Uses price touching weekly Bollinger bands as reversal signals when price is near weekly extremes.
# Works in both bull/bear markets by fading extremes in ranging markets and catching breakouts in trending markets.
# Low trade frequency (<25/year) minimizes fee drag while capturing high-probability moves.

name = "1d_Touchstone_Signal"
timeframe = "1d"
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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    # Calculate weekly Donchian high/low
    wh_max = pd.Series(wh).rolling(window=20, min_periods=20).max().values
    wl_min = pd.Series(wl).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    wh_max_aligned = align_htf_to_ltf(prices, df_1w, wh_max)
    wl_min_aligned = align_htf_to_ltf(prices, df_1w, wl_min)
    
    # Daily Bollinger Bands (20-period, 2 std)
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma20 + (2 * std20)
    lower_bb = ma20 - (2 * std20)
    
    # Volume filter: volume > 1.3x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(wh_max_aligned[i]) or np.isnan(wl_min_aligned[i]) or
            np.isnan(ma20[i]) or np.isnan(std20[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches lower Bollinger Band AND above weekly low (support)
            if close[i] <= lower_bb[i] and close[i] > wl_min_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches upper Bollinger Band AND below weekly high (resistance)
            elif close[i] >= upper_bb[i] and close[i] < wh_max_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches upper Bollinger Band OR breaks below weekly low
            if close[i] >= upper_bb[i] or close[i] <= wl_min_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches lower Bollinger Band OR breaks above weekly high
            if close[i] <= lower_bb[i] or close[i] >= wh_max_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals