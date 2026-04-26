#!/usr/bin/env python3
"""
6h_ElderRay_Trend_VolumeConfirm
Hypothesis: Elder Ray (Bull/Bear Power) combined with 1d EMA trend and volume confirmation captures institutional moves in both bull and bear markets. Bull Power = High - EMA13, Bear Power = Low - EMA13. Long when Bull Power > 0 and rising, Bear Power < 0, price > 1d EMA50, volume > 1.5x average. Short when Bear Power < 0 and falling, Bull Power < 0, price < 1d EMA50, volume > 1.5x average. Uses discrete sizing (0.25) to limit fee churn. Target 12-30 trades/year to avoid fee drag on 6h timeframe.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # EMA13 for Elder Ray (using 6h close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Smooth Bull/Bear Power to reduce noise (2-period EMA)
    bull_power_smooth = pd.Series(bull_power).ewm(span=2, adjust=False, min_periods=2).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=2, adjust=False, min_periods=2).mean().values
    
    # Average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Warmup: max of EMA13(13), EMA50(50), volume(20)
    start_idx = max(13, 50, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_50_val = ema_50_1d_aligned[i]
        bull_val = bull_power_smooth[i]
        bear_val = bear_power_smooth[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_50_val) or np.isnan(avg_vol) or np.isnan(bull_val) or 
            np.isnan(bear_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Rising Bull Power: current > previous
        bull_rising = i > start_idx and bull_val > bull_power_smooth[i-1]
        # Falling Bear Power: current < previous
        bear_falling = i > start_idx and bear_val < bear_power_smooth[i-1]
        
        # Long: Bull Power > 0 and rising, Bear Power < 0, price > 1d EMA50, volume confirmed
        long_condition = (bull_val > 0) and bull_rising and (bear_val < 0) and (close_val > ema_50_val) and volume_confirmed
        # Short: Bear Power < 0 and falling, Bull Power < 0, price < 1d EMA50, volume confirmed
        short_condition = (bear_val < 0) and bear_falling and (bull_val < 0) and (close_val < ema_50_val) and volume_confirmed
        
        # Exit: Elder Ray divergence or trend change
        long_exit = (position == 1 and (bull_val <= 0 or close_val < ema_50_val))
        short_exit = (position == -1 and (bear_val >= 0 or close_val > ema_50_val))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_ElderRay_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0