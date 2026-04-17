#!/usr/bin/env python3
"""
6h 1-Day SuperTrend with Volume Confirmation
Enters long when price closes above SuperTrend (ATR=10, multiplier=3) on daily timeframe with volume > 1.5x 20-period average
Enters short when price closes below SuperTrend with volume > 1.5x 20-period average
Exits when price closes back below/above SuperTrend respectively.
SuperTrend adapts to volatility and trend strength, working in both bull and bear markets.
Target: 20-40 trades/year (80-160 total over 4 years) by requiring SuperTrend signal and volume confirmation.
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
    
    # === Daily SuperTrend (ATR=10, multiplier=3) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range (TR)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate ATR using Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 10
    atr_1d = wilders_smoothing(tr, period)
    
    # Calculate basic upper and lower bands
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + 3 * atr_1d
    lower_band = hl2 - 3 * atr_1d
    
    # Initialize SuperTrend
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    supertrend[period-1] = upper_band[period-1]
    direction[period-1] = 1
    
    for i in range(period, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            # Potential uptrend
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            direction[i] = 1
        else:
            # Potential downtrend
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            direction[i] = -1
    
    # Align SuperTrend and direction to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # === Daily Volume Spike (1.5x 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for all calculations
    warmup = 40
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_today_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_today_aligned[i] > vol_ma_20_1d_aligned[i] * 1.5
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price closes above SuperTrend (uptrend) with volume confirmation
            if close[i] > supertrend_aligned[i] and direction_aligned[i] == 1 and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price closes below SuperTrend (downtrend) with volume confirmation
            elif close[i] < supertrend_aligned[i] and direction_aligned[i] == -1 and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price closes back below/above SuperTrend
        elif position == 1:
            # Exit long: price closes below SuperTrend
            if close[i] < supertrend_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above SuperTrend
            if close[i] > supertrend_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dSuperTrend10x3_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0