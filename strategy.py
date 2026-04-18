#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d ADX Trend Filter
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength.
# Combined with 1d ADX > 25 to filter for trending markets only, avoiding whipsaws in ranges.
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, ADX > 25.
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, ADX > 25.
# Works in bull markets (strong bull power) and bear markets (strong bear power).
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drain.
name = "6h_ElderRay_ADX25_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = ema13 - low   # Bear Power: EMA13 - Low
    
    # Calculate ADX on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    atr_period = 14
    atr_1d = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # ADX (smoothed DX)
    adx_1d = wilders_smoothing(dx, atr_period)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Slope of Elder Ray powers (1-period change)
    bull_power_slope = bull_power - np.roll(bull_power, 1)
    bear_power_slope = bear_power - np.roll(bear_power, 1)
    bull_power_slope[0] = 0
    bear_power_slope[0] = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_slope[i]) or np.isnan(bear_power_slope[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        bp = bull_power[i]
        bp_slope = bull_power_slope[i]
        br = bear_power[i]
        br_slope = bear_power_slope[i]
        adx = adx_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 and rising, Bear Power < 0, ADX > 25
            if bp > 0 and bp_slope > 0 and br < 0 and adx > 25:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling, Bull Power > 0, ADX > 25
            elif br < 0 and br_slope < 0 and bp > 0 and adx > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative or ADX weakens
            if bp <= 0 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns positive or ADX weakens
            if br >= 0 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals