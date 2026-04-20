#!/usr/bin/env python3
# 12h_2w_Fibonacci_Retracement_Trend
# Hypothesis: In trending markets (ADX > 25), price respects 1-week Fibonacci retracement levels (38.2%, 61.8%) from the prior 2-week swing.
# We buy at the 61.8% retracement of an uptrend and sell at the 38.2% retracement of a downtrend using 12-hour closes.
# Volume confirmation filters weak moves. Targets 15-30 trades/year by requiring confluence of Fib level, volume surge, and trend.
# Works in both bull and bear markets due to trend filtering.

name = "12h_2w_Fibonacci_Retracement_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 2-week swing high/low for Fibonacci
    # Use rolling 2-week window (10 trading days approx) for swing points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 2-week rolling max/min for swing points
    lookback = 10  # ~2 weeks of 5 trading days each
    swing_high = pd.Series(high_1w).rolling(window=lookback, min_periods=lookback).max().values
    swing_low = pd.Series(low_1w).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate Fibonacci levels: 38.2% and 61.8% retracements
    diff = swing_high - swing_low
    fib_382 = swing_high - (diff * 0.382)  # 38.2% retracement from high
    fib_618 = swing_high - (diff * 0.618)  # 61.8% retracement from high
    
    # Align weekly Fib levels to 12h timeframe
    fib_382_aligned = align_htf_to_ltf(prices, df_1w, fib_382)
    fib_618_aligned = align_htf_to_ltf(prices, df_1w, fib_618)
    
    # Calculate 1w ADX for trend filter (14-period)
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR and DM using Wilder smoothing
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(fib_382_aligned[i]) or np.isnan(fib_618_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Only trade in trending markets (ADX > 25)
            if adx_aligned[i] > 25:
                # Long at 61.8% retracement (support in uptrend) with volume confirmation
                if (close[i] > fib_618_aligned[i] * 0.995 and 
                    volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
                # Short at 38.2% retracement (resistance in downtrend) with volume
                elif (close[i] < fib_382_aligned[i] * 1.005 and 
                      volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: break below 38.2% level or trend weakening
            if (close[i] < fib_382_aligned[i] * 0.995) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above 61.8% level or trend weakening
            if (close[i] > fib_618_aligned[i] * 1.005) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals