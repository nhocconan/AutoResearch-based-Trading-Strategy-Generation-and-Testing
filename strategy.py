#!/usr/bin/env python3
"""
4H_1D_HullTrend_PriceAction_12HVol
Hypothesis: Price action aligned with Hull MA trend (1D) and confirmed by 12h volume spikes.
Works in both bull and bear: long when price > Hull MA + volume spike, short when price < Hull MA + volume spike.
Target: 30-40 trades/year on 4h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def wma(values, window):
    """Weighted Moving Average."""
    if len(values) < window:
        return np.full_like(values, np.nan, dtype=float)
    weights = np.arange(1, window + 1)
    return np.convolve(values, weights[::-1], mode='full')[:len(values)] / weights.sum()

def hull_moving_average(close, period):
    """Hull Moving Average."""
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    raw_hull = 2 * wma_half - wma_full
    hull = wma(raw_hull, sqrt_period)
    
    return hull

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Hull MA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1D Hull MA (16-period)
    hull_ma_1d = hull_moving_average(df_1d['close'].values, 16)
    hull_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, hull_ma_1d)
    
    # Get 12H data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12H volume average (20-period)
    vol_ma_12h = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Hull MA and volume MA
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(hull_ma_1d_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        hull_trend = hull_ma_1d_aligned[i]
        vol_ma = vol_ma_12h_aligned[i]
        vol_spike = volume[i] > (vol_ma * 2.0)  # Volume spike: 2x average
        
        if position == 0:
            # Long: price > Hull MA + volume spike
            if close[i] > hull_trend and vol_spike:
                signals[i] = size
                position = 1
            # Short: price < Hull MA + volume spike
            elif close[i] < hull_trend and vol_spike:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Hull MA
            if close[i] < hull_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Hull MA
            if close[i] > hull_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4H_1D_HullTrend_PriceAction_12HVol"
timeframe = "4h"
leverage = 1.0