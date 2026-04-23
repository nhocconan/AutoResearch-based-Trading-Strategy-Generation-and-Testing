#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above 4h Donchian upper band (20-period high) AND 12h close > 12h EMA50 (uptrend) AND volume > 2.0x 20-period MA.
Short when price breaks below 4h Donchian lower band (20-period low) AND 12h close < 12h EMA50 (downtrend) AND volume > 2.0x 20-period MA.
Exit when price retraces to the midpoint of the Donchian channel or 12h trend reverses.
Designed for low trade frequency (target: 25-40/year) to minimize fee drag and work in both bull and bear markets via trend filter.
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
    
    # Calculate 4h Donchian channel (20-period)
    lookback = 20
    high_lookback = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    low_lookback = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_upper = high_lookback
    donchian_lower = low_lookback
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_lookback[i]) or np.isnan(low_lookback[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h close > EMA50 = uptrend, close < EMA50 = downtrend
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        trend_up = close_12h_aligned[i] > ema_50_12h_aligned[i]
        trend_down = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        # Volume filter: 4h volume > 2.0x 20-period MA
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND uptrend AND volume filter
            if close[i] > donchian_upper[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND downtrend AND volume filter
            elif close[i] < donchian_lower[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price retraces to Donchian midpoint OR 12h trend turns down
                if close[i] <= donchian_mid[i] or not trend_up:
                    exit_signal = True
            elif position == -1:
                # Short exit: price retraces to Donchian midpoint OR 12h trend turns up
                if close[i] >= donchian_mid[i] or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0