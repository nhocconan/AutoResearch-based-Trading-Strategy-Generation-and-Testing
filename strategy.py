#!/usr/bin/env python3
"""
4h_Fibonacci_Retracement_Breakout_Trend_Filter_v1
Hypothesis: Uses 4h Fibonacci retracement levels (38.2%, 61.8%) calculated from 1d swing high/low.
Enters long when price breaks above 61.8% retracement with volume confirmation and above 1d EMA50 trend.
Enters short when price breaks below 38.2% retracement with volume confirmation and below 1d EMA50 trend.
Uses 1d ADX > 25 to filter for trending markets only. Designed for low trade frequency by requiring
multiple confluences: Fibonacci level break, volume spike, and trend alignment.
Works in bull markets by catching continuation breakouts and in bear markets by selling retracements.
Targets 20-40 trades/year to minimize fee drag.
"""

name = "4h_Fibonacci_Retracement_Breakout_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for swing points, trend, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Swing High and Low (50-period lookback) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate rolling max/min for swing points
    roll_max = pd.Series(high_1d).rolling(window=50, min_periods=1).max().values
    roll_min = pd.Series(low_1d).rolling(window=50, min_periods=1).min().values
    
    # Fibonacci levels: 61.8% and 38.2% retracement
    diff = roll_max - roll_min
    fib_618 = roll_max - 0.618 * diff  # 61.8% retracement level
    fib_382 = roll_max - 0.382 * diff  # 38.2% retracement level
    
    # Align Fibonacci levels to 4h
    fib_618_aligned = align_htf_to_ltf(prices, df_1d, fib_618)
    fib_382_aligned = align_htf_to_ltf(prices, df_1d, fib_382)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # --- 1d Trend Filter (EMA50) ---
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # --- 1d ADX for trend strength filter ---
    # Calculate +DI, -DI, and ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI values
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.nan_to_num(dx, nan=0)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(fib_618_aligned[i]) or np.isnan(fib_382_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(adx_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        # Trend strength filter
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above 61.8% Fib with volume, above EMA50, in trend
            if (close[i] > fib_618_aligned[i] and 
                volume_spike and 
                close[i] > ema_50_aligned[i] and
                trending):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 38.2% Fib with volume, below EMA50, in trend
            elif (close[i] < fib_382_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_50_aligned[i] and
                  trending):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Fib break or loss of trend
            if position == 1:
                # Exit long: price breaks below 38.2% Fib or loss of trend
                if (close[i] < fib_382_aligned[i] or not trending):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above 61.8% Fib or loss of trend
                if (close[i] > fib_618_aligned[i] or not trending):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals