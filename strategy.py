#!/usr/bin/env python3
"""
6h_Donchian_Breakout_WeeklyTrend_VolumeConfirmation
Hypothesis: Donchian(20) breakouts aligned with weekly trend (price > 1w EMA50) and volume confirmation capture high-probability moves. 
Weekly trend filter avoids counter-trend trades. Volume ensures momentum. Designed for 6h timeframe with 1w trend filter.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align all indicators to primary timeframe (6h)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need weekly EMA50 (50), volume avg (20), Donchian (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_confirm_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema50 = ema50_1w_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        
        if position == 0:
            # Long when price breaks above Donchian high with weekly uptrend and volume
            if close_val > ema50 and close_val > upper and vol_conf:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short when price breaks below Donchian low with weekly downtrend and volume
            elif close_val < ema50 and close_val < lower and vol_conf:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit: price re-enters Donchian channel or weekly trend reverses
            if close_val < upper and close_val > lower:
                signals[i] = 0.0
                position = 0
            elif close_val < ema50:  # Weekly trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price re-enters Donchian channel or weekly trend reverses
            if close_val < upper and close_val > lower:
                signals[i] = 0.0
                position = 0
            elif close_val > ema50:  # Weekly trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_Breakout_WeeklyTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0