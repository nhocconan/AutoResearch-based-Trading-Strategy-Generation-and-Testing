#!/usr/bin/env python3
"""
6h_ElderRay_Breakout_1dTrend_VolumeFilter
Hypothesis: On 6h timeframe, use Elder Ray Index (Bull Power/Bear Power) from 1d for breakout confirmation with 1d EMA50 trend filter and volume spike (>2.0x 20-period average). Enter long when Bull Power > 0 and price breaks above 1d EMA50 + volume spike; enter short when Bear Power < 0 and price breaks below 1d EMA50 + volume filter. Exit on opposite signal or trend reversal. Designed for 12-30 trades/year on 6h by requiring strong 1d momentum alignment, reducing fee drag while capturing sustained moves in both bull and bear regimes.
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
    
    # Get 1d data for Elder Ray calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 periods for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA13 and EMA50 for Elder Ray
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 for Bull/Bear Power calculation
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # EMA50 for trend filter
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align 1d indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average on 6h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA50 (50) + volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend alignment from EMA50
        price_above_ema50 = close[i] > ema50_1d_aligned[i]
        price_below_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 + price above 1d EMA50 + volume spike
            long_signal = (bull_power_1d_aligned[i] > 0) and price_above_ema50 and volume_spike[i]
            
            # Short: Bear Power < 0 + price below 1d EMA50 + volume spike
            short_signal = (bear_power_1d_aligned[i] < 0) and price_below_ema50 and volume_spike[i]
            
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
            # Exit: Bear Power < 0 OR price below 1d EMA50
            if (bear_power_1d_aligned[i] < 0 or not price_above_ema50):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bull Power > 0 OR price above 1d EMA50
            if (bull_power_1d_aligned[i] > 0 or not price_below_ema50):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_Breakout_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0