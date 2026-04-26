#!/usr/bin/env python3
"""
6h_ElderRay_ZeroCross_12hTrend_VolumeFilter_v1
Hypothesis: Use Elder Ray Bull/Bear Power zero cross for entry timing with 12h EMA50 trend filter and volume confirmation (>1.5x 20-period average). Elder Ray measures bull/bear power relative to EMA13, providing early reversal signals. Combined with 12h trend to avoid counter-trend trades and volume to confirm conviction. Target: 12-37 trades/year on 6h timeframe. Works in both bull and bear markets by following the 12h trend while using Elder Ray for precise entries.
"""

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need at least 50 periods for EMA50
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(df_12h['close'])
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA13, EMA50_12h, volume MA
    start_idx = max(13, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 12h trend alignment
        trend_12h_uptrend = close[i] > ema_50_12h_aligned[i]
        trend_12h_downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: Bull Power crosses above zero + 12h uptrend + volume spike
            # Bull Power zero cross: previous <= 0 and current > 0
            bull_power_cross_up = (bull_power[i-1] <= 0) and (bull_power[i] > 0)
            long_signal = bull_power_cross_up and trend_12h_uptrend and volume_spike[i]
            
            # Short: Bear Power crosses below zero + 12h downtrend + volume spike
            # Bear Power zero cross: previous >= 0 and current < 0
            bear_power_cross_down = (bear_power[i-1] >= 0) and (bear_power[i] < 0)
            short_signal = bear_power_cross_down and trend_12h_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bear Power crosses below zero OR 12h trend turns down
            bear_power_cross_down = (bear_power[i-1] >= 0) and (bear_power[i] < 0)
            if bear_power_cross_down or not trend_12h_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bull Power crosses above zero OR 12h trend turns up
            bull_power_cross_up = (bull_power[i-1] <= 0) and (bull_power[i] > 0)
            if bull_power_cross_up or not trend_12h_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_ZeroCross_12hTrend_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0