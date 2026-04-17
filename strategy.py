#!/usr/bin/env python3
"""
6h_AdaptiveRegime_FibonacciBreakout
Hypothesis: Adaptive regime detection using 12h ADX and RSI mean-reversion vs trend following.
In trending markets (ADX>25): Break Fibonacci extension levels with volume confirmation.
In ranging markets (ADX<=25): Mean revert at Fibonacci retracement levels.
Uses 1d Fibonacci levels from swing high/low for structure.
Designed for low trade frequency (12-37/year) to work in both bull and bear markets.
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
    
    # === 12h ADX for regime detection ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    tr = np.maximum(high_12h[1:] - low_12h[1:], 
                    np.maximum(np.abs(high_12h[1:] - close_12h[:-1]), 
                               np.abs(low_12h[1:] - close_12h[:-1])))
    
    # Pad arrays to match length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    period = 14
    tr14 = wilders_smoothing(tr, period)
    plus_dm14 = wilders_smoothing(plus_dm, period)
    minus_dm14 = wilders_smoothing(minus_dm, period)
    
    plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
    adx = wilders_smoothing(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 1d Fibonacci levels from swing points ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Find swing high and low over 50-period lookback
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    swing_high = rolling_max(high_1d, 50)
    swing_low = rolling_min(low_1d, 50)
    
    # Calculate Fibonacci levels
    diff = swing_high - swing_low
    fib_0 = swing_low
    fib_236 = swing_low + diff * 0.236
    fib_382 = swing_low + diff * 0.382
    fib_500 = swing_low + diff * 0.5
    fib_618 = swing_low + diff * 0.618
    fib_786 = swing_low + diff * 0.786
    fib_100 = swing_high
    fib_1272 = swing_high + diff * 0.272
    fib_1618 = swing_high + diff * 0.618
    
    # Align Fibonacci levels
    fib_levels = {
        '0': fib_0, '236': fib_236, '382': fib_382, '500': fib_500,
        '618': fib_618, '786': fib_786, '100': fib_100, '1272': fib_1272, '1618': fib_1618
    }
    
    fib_aligned = {}
    for key, values in fib_levels.items():
        fib_aligned[key] = align_htf_to_ltf(prices, df_1d, values)
    
    # === 1d Volume confirmation ===
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    vol_1d_current_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100  # Covers ADX and rolling calculations
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(fib_aligned['382'][i]) or np.isnan(fib_aligned['618'][i]) or
            np.isnan(fib_aligned['100'][i]) or np.isnan(fib_aligned['1272'][i]) or
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(vol_1d_current_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: current volume > 1.3x 20-day average
        vol_filter = vol_1d_current_aligned[i] > 1.3 * vol_avg_20_aligned[i]
        
        # Regime filter: ADX > 25 = trending, ADX <= 25 = ranging
        is_trending = adx_aligned[i] > 25
        
        if position == 0:
            if is_trending:
                # Trending market: Breakout at Fibonacci extensions
                if close[i] > fib_aligned['1272'][i] and vol_filter:
                    signals[i] = 0.25
                    position = 1
                    continue
                elif close[i] < fib_aligned['236'][i] and vol_filter:
                    signals[i] = -0.25
                    position = -1
                    continue
            else:
                # Ranging market: Mean reversion at Fibonacci retracements
                if close[i] < fib_aligned['382'][i] and vol_filter:
                    signals[i] = 0.25
                    position = 1
                    continue
                elif close[i] > fib_aligned['618'][i] and vol_filter:
                    signals[i] = -0.25
                    position = -1
                    continue
        elif position == 1:
            # Long exit conditions
            if is_trending:
                # Exit trend trade at 1618 extension or reversal below 100
                if close[i] > fib_aligned['1618'][i] or close[i] < fib_aligned['100'][i]:
                    signals[i] = 0.0
                    position = 0
                    continue
                else:
                    signals[i] = 0.25
            else:
                # Exit range trade at opposite fib level or midpoint
                if close[i] > fib_aligned['618'][i] or close[i] < fib_aligned['500'][i]:
                    signals[i] = 0.0
                    position = 0
                    continue
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit conditions
            if is_trending:
                # Exit trend trade at 236 retracement or reversal above 100
                if close[i] < fib_aligned['236'][i] or close[i] > fib_aligned['100'][i]:
                    signals[i] = 0.0
                    position = 0
                    continue
                else:
                    signals[i] = -0.25
            else:
                # Exit range trade at opposite fib level or midpoint
                if close[i] < fib_aligned['382'][i] or close[i] > fib_aligned['500'][i]:
                    signals[i] = 0.0
                    position = 0
                    continue
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_AdaptiveRegime_FibonacciBreakout"
timeframe = "6h"
leverage = 1.0