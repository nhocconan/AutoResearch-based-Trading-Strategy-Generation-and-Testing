#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_TrendFilter_v1
Hypothesis: Donchian(20) breakouts on 6h aligned with weekly pivot direction (price vs weekly Camarilla pivot) and 1d EMA50 trend filter capture high-probability moves in both bull and bear markets. Weekly pivot provides structural support/resistance, EMA50 filters counter-trend trades, and Donchian breakouts capture momentum. Discrete sizing (0.25) minimizes fee drag. Target: 75-150 total trades over 4 years.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1w data for weekly pivot (Camarilla) and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    range_1w = high_1w - low_1w
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0  # Standard pivot point
    weekly_r1 = 2 * weekly_pivot - low_1w
    weekly_s1 = 2 * weekly_pivot - high_1w
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align all indicators to primary timeframe (6h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)  # Using 1d as reference for alignment stability
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need EMA50 (50), Donchian (20), weekly pivot (1)
    start_idx = max(50, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema50 = ema50_1d_aligned[i]
        pivot = weekly_pivot_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        dch_high = donchian_high_aligned[i]
        dch_low = donchian_low_aligned[i]
        
        if position == 0:
            # Determine trend alignment: price vs weekly pivot and 1d EMA50
            # Bullish: price above weekly pivot AND above EMA50
            # Bearish: price below weekly pivot AND below EMA50
            bullish = close_val > pivot and close_val > ema50
            bearish = close_val < pivot and close_val < ema50
            
            if bullish and vol_conf:
                # Long when price breaks above Donchian high in bullish alignment
                if close_val > dch_high:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif bearish and vol_conf:
                # Short when price breaks below Donchian low in bearish alignment
                if close_val < dch_low:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions: stoploss (2.0*ATR) or Donchian low touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price - 2.0 * atr_approx
            
            if close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < dch_low:  # Donchian low touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: stoploss (2.0*ATR) or Donchian high touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price + 2.0 * atr_approx
            
            if close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val > dch_high:  # Donchian high touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0