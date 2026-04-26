#!/usr/bin/env python3
"""
6h_ElderRay_ZeroCross_12hTrend_VolumeFilter
Hypothesis: Use 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) zero-cross signals filtered by 12h trend (close > EMA50) and volume (>1.5x 20-period average). Elder Ray captures momentum shifts; 12h trend ensures alignment with higher timeframe direction; volume filter confirms institutional interest. Works in bull/bear by following trend. Target: 12-35 trades/year (50-150 over 4 years). Discrete sizing: ±0.25.
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
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Zero-cross signals: Bull Power crosses above 0 (long), Bear Power crosses below 0 (short)
    bull_cross_up = (bull_power[1:] > 0) & (bull_power[:-1] <= 0)
    bear_cross_down = (bear_power[1:] < 0) & (bear_power[:-1] >= 0)
    # Prepend False for first bar
    bull_cross_up = np.concatenate([[False], bull_cross_up])
    bear_cross_down = np.concatenate([[False], bear_cross_down])
    
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
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma[i]):
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
            # Long: Bull Power crosses above 0 + 12h uptrend + volume spike
            long_signal = bull_cross_up[i] and trend_12h_uptrend and volume_spike[i]
            
            # Short: Bear Power crosses below 0 + 12h downtrend + volume spike
            short_signal = bear_cross_down[i] and trend_12h_downtrend and volume_spike[i]
            
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
            # Exit: Bull Power crosses below 0 OR 12h trend turns down
            if bull_power[i] < 0 or not trend_12h_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bear Power crosses above 0 OR 12h trend turns up
            if bear_power[i] > 0 or not trend_12h_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_ZeroCross_12hTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0