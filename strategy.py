#!/usr/bin/env python3
"""
6h_ElderRay_ZeroCross_12hTrend_VolumeFilter_v1
Hypothesis: Elder Ray (Bull/Bear Power) zero-cross signals filtered by 12h EMA50 trend and volume spike. Works in bull/bear via trend filter; Elder Ray measures bull/bear strength relative to EMA13. Discrete size 0.25 limits fee drag. Target 12-30 trades/year.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Zero-cross signals: Bull Power crosses above 0 (long), Bear Power crosses below 0 (short)
    bull_cross_up = (bull_power > 0) & (np.roll(bull_power, 1) <= 0)
    bear_cross_down = (bear_power < 0) & (np.roll(bear_power, 1) >= 0)
    bull_cross_up[0] = False
    bear_cross_down[0] = False
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.8x 30-period average
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA13 (13), 12h EMA50 (50), volume MA (30)
    start_idx = max(13, 50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma[i])):
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
            # Long: Bull Power crosses above 0 + volume spike + 12h uptrend
            long_signal = bull_cross_up[i] and volume_spike[i] and trend_12h_uptrend
            
            # Short: Bear Power crosses below 0 + volume spike + 12h downtrend
            short_signal = bear_cross_down[i] and volume_spike[i] and trend_12h_downtrend
            
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
            # Exit: Bear Power crosses below 0 OR 12h trend turns down
            if bear_cross_down[i] or not trend_12h_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bull Power crosses above 0 OR 12h trend turns up
            if bull_cross_up[i] or not trend_12h_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_ZeroCross_12hTrend_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0