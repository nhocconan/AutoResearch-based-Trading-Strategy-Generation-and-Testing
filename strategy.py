#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotTrend_12hVolumeSpike_v1
Hypothesis: On 6h timeframe, trade Donchian(20) breakouts aligned with weekly pivot trend (price above/below weekly pivot) and confirmed by 12h volume spike (>1.5x 20-period average). Weekly pivot provides structural support/resistance that works in both bull and bear markets, while volume spike filters false breakouts. Target 12-30 trades/year by requiring confluence of weekly trend alignment, price structure breakout, and volume confirmation. Designed to avoid overtrading and work in ranging markets by only trading when price is trending relative to weekly pivot.
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
    
    # Get 6h data for Donchian calculation (we need 20 periods)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 6h: upper=20-period high, lower=20-period low
    high_6h = pd.Series(df_6h['high'].values)
    low_6h = pd.Series(df_6h['low'].values)
    donch_upper_6h = high_6h.rolling(window=20, min_periods=20).max().values
    donch_lower_6h = low_6h.rolling(window=20, min_periods=20).min().values
    
    # Get weekly data for pivot trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h volume spike: current volume > 1.5x 20-period average
    vol_12h = pd.Series(df_12h['volume'].values)
    vol_ma_20_12h = vol_12h.rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = vol_12h.values > (1.5 * vol_ma_20_12h)
    
    # Align all HTF indicators to 6h timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_6h, donch_upper_6h)
    donch_lower_aligned = align_htf_to_ltf(prices, df_6h, donch_lower_6h)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Donchian(20) 6h, weekly pivot (need 1 bar), volume MA(20) 12h
    start_idx = max(20, 1, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_upper_aligned[i]) or 
            np.isnan(donch_lower_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        donch_upper = donch_upper_aligned[i]
        donch_lower = donch_lower_aligned[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        weekly_r1_val = weekly_r1_aligned[i]
        weekly_s1_val = weekly_s1_aligned[i]
        volume_spike = bool(volume_spike_aligned[i])
        close_val = close[i]
        
        # Trend filter: price above/below weekly pivot
        above_weekly_pivot = close_val > weekly_pivot_val
        below_weekly_pivot = close_val < weekly_pivot_val
        
        if position == 0:
            # Long: break above Donchian upper with above weekly pivot and volume spike
            long_signal = (close_val > donch_upper) and \
                          above_weekly_pivot and \
                          volume_spike
            
            # Short: break below Donchian lower with below weekly pivot and volume spike
            short_signal = (close_val < donch_lower) and \
                           below_weekly_pivot and \
                           volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long: exit when price crosses below weekly pivot or Donchian lower
            signals[i] = 0.25
            if close_val < weekly_pivot_val or close_val < donch_lower:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short: exit when price crosses above weekly pivot or Donchian upper
            signals[i] = -0.25
            if close_val > weekly_pivot_val or close_val > donch_upper:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotTrend_12hVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0