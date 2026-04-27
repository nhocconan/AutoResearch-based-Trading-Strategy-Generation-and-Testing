#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirm
Hypothesis: Ichimoku cloud breakout on 6h with weekly trend filter and volume confirmation.
Long when price breaks above Kumo cloud with weekly bullish trend (price > weekly Kumo) and volume spike.
Short when price breaks below Kumo cloud with weekly bearish trend (price < weekly Kumo) and volume spike.
Uses cloud as dynamic support/resistance and weekly trend for higher timeframe alignment.
Volume spike confirms institutional participation. Designed for 12-30 trades/year on 6h to minimize fee drag.
Works in both bull and bear markets by following weekly trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Kumo cloud boundaries: Senkou Span A and B shifted forward 26 periods
    # For backtesting, we use current cloud (no look-ahead)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Weekly trend: price > weekly Kumo (bullish) or price < weekly Kumo (bearish)
    # Calculate weekly Ichimoku cloud
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # Weekly Tenkan-sen (9-period)
    wk_period9_high = pd.Series(df_1w_high).rolling(window=9, min_periods=9).max().values
    wk_period9_low = pd.Series(df_1w_low).rolling(window=9, min_periods=9).min().values
    wk_tenkan = (wk_period9_high + wk_period9_low) / 2
    
    # Weekly Kijun-sen (26-period)
    wk_period26_high = pd.Series(df_1w_high).rolling(window=26, min_periods=26).max().values
    wk_period26_low = pd.Series(df_1w_low).rolling(window=26, min_periods=26).min().values
    wk_kijun = (wk_period26_high + wk_period26_low) / 2
    
    # Weekly Senkou Span A and B
    wk_senkou_a = (wk_tenkan + wk_kijun) / 2
    wk_period52_high = pd.Series(df_1w_high).rolling(window=52, min_periods=52).max().values
    wk_period52_low = pd.Series(df_1w_low).rolling(window=52, min_periods=52).min().values
    wk_senkou_b = (wk_period52_high + wk_period52_low) / 2
    
    # Weekly Kumo cloud (current values, no shift for trend filter)
    wk_kumo_top = np.maximum(wk_senkou_a, wk_senkou_b)
    wk_kumo_bottom = np.minimum(wk_senkou_a, wk_senkou_b)
    
    # Align weekly Kumo to 6h timeframe
    wk_kumo_top_aligned = align_htf_to_ltf(prices, df_1w, wk_kumo_top)
    wk_kumo_bottom_aligned = align_htf_to_ltf(prices, df_1w, wk_kumo_bottom)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for Ichimoku calculations (52 periods) + weekly alignment + volume
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i]) or
            np.isnan(wk_kumo_top_aligned[i]) or np.isnan(wk_kumo_bottom_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        wk_kumo_top_val = wk_kumo_top_aligned[i]
        wk_kumo_bottom_val = wk_kumo_bottom_aligned[i]
        vol_spike = volume_spike[i]
        
        # Kumo cloud boundaries (current)
        cloud_top = max(senkou_a_shifted[i], senkou_b_shifted[i])
        cloud_bottom = min(senkou_a_shifted[i], senkou_b_shifted[i])
        
        if position == 0:
            # Flat - look for entry: Kumo breakout with weekly trend alignment and volume spike
            # Long: Price breaks above cloud TOP AND weekly bullish trend (price > weekly Kumo top) AND volume spike
            # Short: Price breaks below cloud BOTTOM AND weekly bearish trend (price < weekly Kumo bottom) AND volume spike
            long_condition = (close_val > cloud_top and 
                            close_val > wk_kumo_top_val and 
                            vol_spike)
            short_condition = (close_val < cloud_bottom and 
                             close_val < wk_kumo_bottom_val and 
                             vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below cloud bottom OR weekly trend turns bearish
            if close_val < cloud_bottom or close_val < wk_kumo_bottom_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above cloud top OR weekly trend turns bullish
            if close_val > cloud_top or close_val > wk_kumo_top_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0