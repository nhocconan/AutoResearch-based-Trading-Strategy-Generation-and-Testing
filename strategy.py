#!/usr/bin/env python3
"""
4h 38.2% Fibonacci Retracement Pullback with 1d ADX Trend Filter and Volume Confirmation
Hypothesis: In trending markets, price often pulls back to the 38.2% Fibonacci level of the recent swing.
Buying near this level in an uptrend (or selling in a downtrend) with volume confirmation offers
high-probability entries. The 1d ADX > 25 ensures we only trade in strong trends, avoiding chop.
Works in bull markets via long pullbacks and in bear markets via short pullbacks. Low trade frequency
due to strict trend and level requirements.
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
    
    # Get 1d data for ADX trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period])  # Skip index 0 (undefined)
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.full_like(dx, np.nan)
    # First ADX is average of first 14 DX values
    if len(dx) >= 28:  # Need 14 for initial + 14 for smoothing
        adx[27] = np.nanmean(dx[14:28])  # Days 14-27 (0-indexed)
        for i in range(28, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14  # Wilder smoothing
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate swing high/low for Fibonacci levels (using 20-period lookback)
    def calculate_swing(high_arr, low_arr, lookback=20):
        swing_high = np.full_like(high_arr, np.nan)
        swing_low = np.full_like(low_arr, np.nan)
        for i in range(lookback, len(high_arr)):
            swing_high[i] = np.max(high_arr[i-lookback:i])
            swing_low[i] = np.min(low_arr[i-lookback:i])
        return swing_high, swing_low
    
    swing_high, swing_low = calculate_swing(high, low, 20)
    
    # Calculate 38.2% Fibonacci retracement level
    diff = swing_high - swing_low
    fib_382 = swing_low + 0.382 * diff  # For uptrend pullbacks
    fib_618 = swing_low + 0.618 * diff  # For downtrend pullbacks (alternative)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(fib_382[i]) or np.isnan(fib_618[i]):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        vol_ok = vol_confirm[i]
        fib_382_level = fib_382[i]
        fib_618_level = fib_618[i]
        swing_high_level = swing_high[i]
        swing_low_level = swing_low[i]
        
        if position == 0:
            # Enter long: pullback to 38.2% in uptrend (ADX > 25 and price above swing low)
            if (adx_val > 25 and 
                vol_ok and 
                close[i] <= fib_382_level * 1.005 and  # Allow small tolerance
                close[i] >= fib_382_level * 0.995 and
                close[i] > swing_low_level):
                signals[i] = 0.25
                position = 1
            # Enter short: pullback to 61.8% in downtrend (ADX > 25 and price below swing high)
            elif (adx_val > 25 and 
                  vol_ok and 
                  close[i] >= fib_618_level * 0.995 and  # Allow small tolerance
                  close[i] <= fib_618_level * 1.005 and
                  close[i] < swing_high_level):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks above swing high (take profit) or trend weakens (ADX < 20)
            if close[i] >= swing_high_level or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks below swing low (take profit) or trend weakens (ADX < 20)
            if close[i] <= swing_low_level or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Fibonacci_Pullback_ADXTrend_Volume"
timeframe = "4h"
leverage = 1.0