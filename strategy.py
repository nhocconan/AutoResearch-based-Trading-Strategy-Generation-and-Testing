#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeFilter_v1
Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 1-day EMA50 trend filter and volume confirmation.
Elder Ray measures bull/bear power relative to EMA13, providing early trend strength signals.
In bull markets: go long when Bull Power > 0 and rising + EMA50 uptrend + volume confirmation.
In bear markets: go short when Bear Power < 0 and falling + EMA50 downtrend + volume confirmation.
Designed for low trade frequency (target 15-25/year) to minimize fee drag while capturing strong trends.
Uses 6h primary timeframe with 1d HTF for trend/Elder Ray smoothing.
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
    
    # Get daily data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on daily for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA(13) on 6h for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Smooth Elder Ray with EMA(8) to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=8, adjust=False, min_periods=8).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Calculate volume ratio (current / 30-period average) for confirmation
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of daily EMA(50), 6h EMA(13), smoothing EMA(8), volume MA(30)
    start_idx = max(50, 13, 8, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(bull_power_smooth[i]) or
            np.isnan(bear_power_smooth[i]) or
            np.isnan(vol_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_confirm = vol_ratio[i] > 1.5  # volume at least 1.5x average
        trend_1d_up = close_val > ema_50_1d_aligned[i]
        trend_1d_down = close_val < ema_50_1d_aligned[i]
        
        # Elder Ray signals with momentum
        bull_rising = bull_power_smooth[i] > bull_power_smooth[i-1]
        bear_falling = bear_power_smooth[i] < bear_power_smooth[i-1]
        
        if position == 0:
            # Long: Bull Power > 0 and rising + 1d uptrend + volume confirmation
            long_signal = (bull_power_smooth[i] > 0) and bull_rising and trend_1d_up and vol_confirm
            
            # Short: Bear Power < 0 and falling + 1d downtrend + volume confirmation
            short_signal = (bear_power_smooth[i] < 0) and bear_falling and trend_1d_down and vol_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: 1d trend flips down OR Bear Power turns positive (bull exhaustion)
            if (not trend_1d_up) or (bear_power_smooth[i] > 0):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: 1d trend flips up OR Bull Power turns negative (bear exhaustion)
            if (not trend_1d_down) or (bull_power_smooth[i] < 0):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0