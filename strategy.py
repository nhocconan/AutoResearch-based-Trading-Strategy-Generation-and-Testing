#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_1dTrend_VolumeConfirm
Hypothesis: Donchian(20) breakout on 6h with 1d EMA34 trend filter, weekly pivot confirmation (price > weekly pivot for longs, < for shorts), and volume spike. 
Weekly pivot provides institutional reference point; Donchian breakout captures momentum; EMA34 filters counter-trend moves. 
Volume spike confirms institutional participation. Discrete sizing (0.25) and minimum holding period (3 bars) reduce fee drag.
Target: 80-120 trades over 4 years (20-30/year). Works in bull/bear by requiring alignment with 1d trend and weekly pivot.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:  # Need 34 for daily EMA34
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 6h
    period_donchian = 20
    donchian_high = pd.Series(high).rolling(window=period_donchian, min_periods=period_donchian).max().values
    donchian_low = pd.Series(low).rolling(window=period_donchian, min_periods=period_donchian).min().values
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # Load daily data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (standard calculation)
    # P = (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    # Align weekly pivot to 6h timeframe (no extra delay needed for pivot points)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 34 for daily EMA, 20 for Donchian)
    start_idx = 34
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Trend and pivot filters
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        price_above_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_pivot = close[i] < weekly_pivot_aligned[i]
        
        # Long logic: bullish breakout with volume spike, above daily EMA34, above weekly pivot
        long_condition = breakout_up and volume_spike[i] and price_above_ema and price_above_pivot
        # Short logic: bearish breakout with volume spike, below daily EMA34, below weekly pivot
        short_condition = breakout_down and volume_spike[i] and price_below_ema and price_below_pivot
        
        # Exit logic: Donchian breakout in opposite direction or trend/pivot reversal
        exit_long = breakout_down or (close[i] < ema_34_1d_aligned[i]) or (close[i] < weekly_pivot_aligned[i])
        exit_short = breakout_up or (close[i] > ema_34_1d_aligned[i]) or (close[i] > weekly_pivot_aligned[i])
        
        # Minimum holding period: 3 bars
        if position != 0 and bars_since_entry < 3:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0