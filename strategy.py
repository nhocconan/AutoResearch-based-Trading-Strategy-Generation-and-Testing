#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d Regime Filter and Volume Spike
- Bollinger Bands (20,2) squeeze when bandwidth < 20th percentile of last 50 periods
- Breakout long when price closes above upper band AND volume > 2x 20-period average
- Breakout short when price closes below lower band AND volume > 2x 20-period average
- 1d ADX > 25 confirms trending regime to avoid false breakouts in ranging markets
- Exit when price reverts to middle band (20-period SMA) or opposite band touch
- Designed to capture explosive moves after low volatility periods in both bull and bear markets
- Signal size: 0.25 discrete levels
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2)
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = ma_20 + (2 * std_20)
    lower_band = ma_20 - (2 * std_20)
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper_band - lower_band) / ma_20
    # Squeeze when BB width is below 20th percentile of last 50 periods
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze_condition = bb_width < bb_width_percentile
    
    # Volume confirmation: volume > 2 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = dm_minus[0] = 0
    
    # Wilder's smoothing
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        if len(values) < period:
            return result
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, period)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smoothed / np.where(tr_smoothed == 0, 1, tr_smoothed)
    di_minus = 100 * dm_minus_smoothed / np.where(tr_smoothed == 0, 1, tr_smoothed)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx_1d = wilders_smoothing(dx, period)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Regime filter: trending if ADX > 25
    trending_regime = adx_1d_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 30) + 1  # Need BB, percentile, and ADX data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ma_20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(squeeze_condition[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: squeeze + close above upper band + volume spike + trending regime
            if (squeeze_condition[i] and close[i] > upper_band[i] and 
                volume_spike[i] and trending_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: squeeze + close below lower band + volume spike + trending regime
            elif (squeeze_condition[i] and close[i] < lower_band[i] and 
                  volume_spike[i] and trending_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to middle band OR touches lower band
            if close[i] <= ma_20[i] or close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to middle band OR touches upper band
            if close[i] >= ma_20[i] or close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BBSqueeze_Breakout_1dADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0