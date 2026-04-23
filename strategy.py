#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 12h ADX trend filter and volume confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
- Long: Bull Power > 0 and rising + Bear Power < 0 + ADX > 25 + volume > 1.5x 20-period avg
- Short: Bear Power < 0 and falling + Bull Power > 0 + ADX > 25 + volume > 1.5x 20-period avg
- Exit: Opposite Elder Ray signal or ADX < 20 (trend weakening)
- Uses Elder Ray for trend strength measurement, ADX for trend filter, volume for conviction
- Works in both bull (buy strength in uptrend) and bear (sell weakness in downtrend) markets
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Elder Ray Index components (EMA13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 12h ADX for trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def wilders_smoothing(values, period):
        """Wilder's smoothing (EMA variant)"""
        result = np.full_like(values, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(values) >= period and not np.isnan(values[period-1]):
            result[period-1] = np.nanmean(values[:period])
        # Subsequent values
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]) and not np.isnan(values[i]):
                result[i] = result[i-1] + alpha * (values[i] - result[i-1])
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 14)  # Need 20 for volume MA, 13 for EMA13, 14+14 for ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
# Elder Ray directional changes (using previous bar to detect actual change)
        bull_power_prev = bull_power[i-1]
        bear_power_prev = bear_power[i-1]
        
        bull_power_rising = bull_power[i] > bull_power_prev
        bear_power_falling = bear_power[i] < bear_power_prev
        
        if position == 0:
            # Long: Bull Power > 0 and rising + Bear Power < 0 + ADX > 25 + volume confirmation
            if (bull_power[i] > 0 and bull_power_rising and 
                bear_power[i] < 0 and 
                adx_aligned[i] > 25 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling + Bull Power > 0 + ADX > 25 + volume confirmation
            elif (bear_power[i] < 0 and bear_power_falling and 
                  bull_power[i] > 0 and 
                  adx_aligned[i] > 25 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR ADX < 20 (trend weakening)
            if bull_power[i] <= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 OR ADX < 20 (trend weakening)
            if bear_power[i] >= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADX_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0