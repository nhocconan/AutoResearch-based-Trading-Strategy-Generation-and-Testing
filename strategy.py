#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above 20-period high AND close > 1d EMA50 (uptrend) AND volume > 1.8x 20-period MA.
Short when price breaks below 20-period low AND close < 1d EMA50 (downtrend) AND volume > 1.8x 20-period MA.
Exit when price returns to the opposite Donchian level (long exits at 20-period low, short exits at 20-period high).
Designed for ~12-30 trades/year on 12h timeframe with structure-based edge that works in both bull and bear markets.
Donchian channels provide clear breakout levels; 1d EMA50 ensures higher timeframe alignment; volume confirms conviction.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period Donchian channels on 12h data
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50, Donchian20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA50 = uptrend, close < 1d EMA50 = downtrend
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: 12h volume > 1.8x 20-period MA
        vol_filter = volume[i] > 1.8 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_ma_20[i]  # Break above 20-period high
        breakout_down = close[i] < low_ma_20[i]  # Break below 20-period low
        exit_long = close[i] < low_ma_20[i]      # Long exit: price returns to 20-period low
        exit_short = close[i] > high_ma_20[i]    # Short exit: price returns to 20-period high
        
        if position == 0:
            # Long: Break above 20-period high AND uptrend AND volume confirmation
            if breakout_up and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below 20-period low AND downtrend AND volume confirmation
            elif breakout_down and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite Donchian level
            exit_signal = False
            if position == 1:
                exit_signal = exit_long
            elif position == -1:
                exit_signal = exit_short
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0